use libc::{c_char, c_int};
use rawler::decoders::{Orientation, RawDecodeParams};
use rawler::imgop::develop::{Intermediate, RawDevelop};
use rawler::rawsource::RawSource;
use rayon::prelude::*;
use resvg::{tiny_skia, usvg};
use std::ffi::CStr;
use std::panic::catch_unwind;
use std::path::Path;
use std::slice;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::UNIX_EPOCH;
use usvg::fontdb;

// FFI bridge to C# (via P/Invoke). All exported functions share the same memory
// contract: Rust allocates via Box<[u8]>, caller frees via free_rust_buffer.
// Panics are caught at the boundary and returned as null rather than unwinding into C#.
//
// Pixel format: all functions return straight (unpremultiplied) RGBA8, matching
// DirectXPixelFormat.R8G8B8A8UIntNormalized on the C# side.

// Font database is initialised once on first SVG render and reused for all
// subsequent calls. Loading system fonts on Windows can take tens to hundreds of
// milliseconds, making per-call initialisation unacceptable.
static FONT_DB: LazyLock<Arc<fontdb::Database>> = LazyLock::new(|| {
    let mut db = fontdb::Database::new();
    db.load_system_fonts();
    Arc::new(db)
});

/// Parsed SVGs kept across zoom re-renders. One photo viewer rarely has more than a
/// couple of SVGs warm at once; the cap just stops a long session from retaining every
/// file it ever opened.
const SVG_CACHE_LIMIT: usize = 4;
/// Safety cap on one raster buffer. This is not a zoom limit: the caller renders only
/// the visible region, which is about the size of the screen. 8192 stops a bad request
/// from allocating a multi-gigabyte pixmap.
const SVG_MAX_EDGE: u32 = 8192;

struct CachedSvg {
    path: String,
    len: u64,
    modified_ms: u128,
    tree: Arc<usvg::Tree>,
}

static SVG_CACHE: Mutex<Vec<CachedSvg>> = Mutex::new(Vec::new());

/// Borrows a C string as a &str for the duration of the calling scope.
/// SAFETY: Caller must ensure `ptr` remains valid for 'a. Lifetime is unconstrained
/// by the type system — do not store the return value beyond the current call frame.
fn ptr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().ok()
}

#[inline(always)]
fn f32_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0) as u8
}

/// Writes `value` to `ptr` only if `ptr` is non-null.
unsafe fn safe_write_int(ptr: *mut c_int, value: c_int) {
    if !ptr.is_null() {
        *ptr = value;
    }
}

/// Fully decodes a RAW file through the complete rawler pipeline
/// (demosaicing, white balance, tone mapping) and returns a heap-allocated
/// RGBA8 buffer. Caller receives width/height/rotation via out-params.
///
/// Returns null on any failure (bad path, unsupported format, decode error, panic).
/// On success, caller MUST free the buffer with `free_rust_buffer(ptr, width*height*4)`.
///
/// Note: `raw_image.orientation` is hardcoded to Normal in rawler 0.7.x, so
/// orientation is read from `raw_metadata` instead.
#[unsafe(no_mangle)]
pub extern "C" fn rawler_render_raw(
    path_ptr: *const c_char,
    width: *mut c_int,
    height: *mut c_int,
    rotation: *mut c_int,
) -> *mut u8 {
    let result = catch_unwind(|| {
        // Inner closure allows us to use the `?` operator safely.
        let process = || -> Option<*mut u8> {
            let path = ptr_to_str(path_ptr)?;

            let rawsource = RawSource::new(Path::new(path)).ok()?;
            let decoder = rawler::get_decoder(&rawsource).ok()?;
            let params = RawDecodeParams::default();

            // `raw_image.orientation` is hardcoded to `Normal` in rawler 0.7.x,
            // so we read orientation from raw_metadata instead.
            let metadata = decoder.raw_metadata(&rawsource, &params).ok()?;
            let orientation_tag = metadata.exif.orientation.unwrap_or(1);
            let rotation_degrees = match Orientation::from_u16(orientation_tag) {
                Orientation::Rotate90 => 90,
                Orientation::Rotate180 => 180,
                Orientation::Rotate270 => 270,
                _ => 0,
            };

            // false = full demosaicing decode.
            let raw_image = decoder.raw_image(&rawsource, &params, false).ok()?;

            let intermediate = RawDevelop::default().develop_intermediate(&raw_image).ok()?;
            let dim = intermediate.dim();
            let w = dim.w;
            let h = dim.h;

            let mut rgba = vec![0u8; w * h * 4];

            // Parallelised over rayon; ThreeColor/FourColor kept as separate arms
            // to allow divergent handling in future.
            match intermediate {
                Intermediate::ThreeColor(img) => {
                    rgba.par_chunks_exact_mut(4)
                        .zip(img.data.par_iter())
                        .for_each(|(out, px)| {
                            out[0] = f32_to_u8(px[0]); // R
                            out[1] = f32_to_u8(px[1]); // G
                            out[2] = f32_to_u8(px[2]); // B
                            out[3] = 255;              // A
                        });
                }
                Intermediate::FourColor(img) => {
                    rgba.par_chunks_exact_mut(4)
                        .zip(img.data.par_iter())
                        .for_each(|(out, px)| {
                            out[0] = f32_to_u8(px[0]); // R
                            out[1] = f32_to_u8(px[1]); // G
                            out[2] = f32_to_u8(px[2]); // B
                            out[3] = 255;              // A
                        });
                }
                Intermediate::Monochrome(img) => {
                    rgba.par_chunks_exact_mut(4)
                        .zip(img.data.par_iter())
                        .for_each(|(out, v)| {
                            let val = f32_to_u8(*v);
                            out[0] = val;
                            out[1] = val;
                            out[2] = val;
                            out[3] = 255;
                        });
                }
            }

            unsafe {
                safe_write_int(width, w as c_int);
                safe_write_int(height, h as c_int);
                safe_write_int(rotation, rotation_degrees);
            }

            Some(Box::into_raw(rgba.into_boxed_slice()) as *mut u8)
        };

        process().unwrap_or(std::ptr::null_mut())
    });

    result.unwrap_or(std::ptr::null_mut())
}

/// Extracts the embedded JPEG preview from a RAW file without full demosaicing.
/// Returns a heap-allocated RGBA8 buffer of the preview image.
/// Also writes the primary RAW dimensions (pw, ph) for aspect-ratio scaling on the C# side.
///
/// Returns null on any failure. On success, caller MUST free with
/// `free_rust_buffer(ptr, width*height*4)`.
#[unsafe(no_mangle)]
pub extern "C" fn rawler_get_embedded_preview_from_raw(
    path_ptr: *const c_char,
    width: *mut c_int,
    height: *mut c_int,
    rotation: *mut c_int,
    primary_width: *mut c_int,
    primary_height: *mut c_int,
) -> *mut u8 {
    let result = catch_unwind(|| {
        let process = || -> Option<*mut u8> {
            let path = ptr_to_str(path_ptr)?;

            let rawsource = RawSource::new(Path::new(path)).ok()?;
            let decoder = rawler::get_decoder(&rawsource).ok()?;
            let params = RawDecodeParams::default();
            let metadata = decoder.raw_metadata(&rawsource, &params).ok()?;

            let orientation_tag = metadata.exif.orientation.unwrap_or(1);
            let rotation_degrees = match Orientation::from_u16(orientation_tag) {
                Orientation::Rotate90 => 90,
                Orientation::Rotate180 => 180,
                Orientation::Rotate270 => 270,
                _ => 0,
            };

            // true = fast thumbnail extraction mode, skips demosaicing.
            let (pw, ph) = match decoder.raw_image(&rawsource, &params, true) {
                Ok(ri) => (ri.width as c_int, ri.height as c_int),
                Err(_) => (0, 0),
            };

            let dynamic_img = rawler::analyze::extract_thumbnail_pixels(path, &params).ok()?;
            
            // Convert any internal format to an RGBA byte buffer efficiently
            let rgba = dynamic_img.into_rgba8();
            let img_w = rgba.width() as c_int;
            let img_h = rgba.height() as c_int;
            let pixels = rgba.into_raw(); // Consumes it into a flat Vec<u8>

            if pixels.is_empty() {
                return None;
            }

            unsafe {
                safe_write_int(width, img_w);
                safe_write_int(height, img_h);
                safe_write_int(rotation, rotation_degrees);
                safe_write_int(primary_width, pw);
                safe_write_int(primary_height, ph);
            }

            Some(Box::into_raw(pixels.into_boxed_slice()) as *mut u8)
        };

        process().unwrap_or(std::ptr::null_mut())
    });

    result.unwrap_or(std::ptr::null_mut())
}

/// Frees a buffer previously returned by `rawler_render_raw`,
/// `rawler_get_embedded_preview_from_raw`, or `resvg_render_svg`.
/// `size` MUST equal width * height * 4 — the exact byte count used at allocation.
/// Passing the wrong size causes Rust to reconstruct a Box with incorrect length,
/// resulting in heap corruption or a double-free.
#[unsafe(no_mangle)]
pub extern "C" fn free_rust_buffer(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        // slice_from_raw_parts_mut constructs the fat pointer directly without
        // creating an intermediate &mut reference, avoiding strict-provenance UB.
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, size));
    }
}

/// Returns a heap-allocated array of null-terminated C strings listing every
/// RAW file extension that rawler can decode (e.g. "ARW\0", "CR2\0", …).
/// The number of entries is written to `*size`.
///
/// All strings are upper-case (as defined by rawler's `SUPPORTED_FILES_EXT`).
///
/// # Memory contract
/// The caller MUST free the returned pointer with `free_formats_buffer(ptr, size)`.
/// Do NOT call `free_rust_buffer` on this pointer — they use different allocation layouts.
#[unsafe(no_mangle)]
pub extern "C" fn rawler_get_supported_formats(size: *mut c_int) -> *mut *mut c_char {
    use rawler::decoders::supported_extensions;
    use std::ffi::CString;

    let exts = supported_extensions();

    // Allocate each extension as an independent CString on the heap.
    let ptrs: Vec<*mut c_char> = exts
        .iter()
        .map(|s| {
            // unwrap: rawler extension strings are pure ASCII, no interior NULs.
            CString::new(*s).unwrap().into_raw()
        })
        .collect();

    let len = ptrs.len() as c_int;

    // Box<[T]> carries its own length in the fat pointer and has no capacity/length
    // mismatch concern — safer than Vec::forget which relies on shrink_to_fit succeeding.
    let boxed: Box<[*mut c_char]> = ptrs.into_boxed_slice();
    let out_ptr = Box::into_raw(boxed) as *mut *mut c_char;

    unsafe {
        safe_write_int(size, len);
    }

    out_ptr
}

/// Frees a pointer previously returned by `rawler_get_supported_formats`.
/// `size` MUST be the same value that was written to the `size` out-param
/// of `rawler_get_supported_formats`.
#[unsafe(no_mangle)]
pub extern "C" fn free_formats_buffer(ptr: *mut *mut c_char, size: c_int) {
    if ptr.is_null() || size <= 0 {
        return;
    }
    unsafe {
        // Reconstruct the boxed slice so Rust drops it correctly.
        let slice = slice::from_raw_parts_mut(ptr, size as usize);
        for p in slice.iter() {
            if !p.is_null() {
                // Reclaim each CString so its memory is freed.
                let _ = std::ffi::CString::from_raw(*p);
            }
        }
        // Drop the outer array using the same Box<[T]> that allocated it.
        let _ = Box::from_raw(slice as *mut [*mut c_char]);
    }
}

/// Writes the SVG's intrinsic size in user units (the viewBox / width / height resvg
/// resolves, not a raster). Returns 1 on success and 0 on failure.
///
/// This is the document size ImageGlass v10 reads from the picture cull rect: 100% zoom
/// is one SVG unit per screen pixel, and any larger zoom re-renders instead of stretching
/// a bitmap.
#[unsafe(no_mangle)]
pub extern "C" fn resvg_svg_intrinsic_size(
    path_ptr: *const c_char,
    width: *mut f32,
    height: *mut f32,
) -> c_int {
    let result = catch_unwind(|| {
        let path = ptr_to_str(path_ptr)?;
        let tree = load_svg_tree(path)?;
        let w = tree.size().width();
        let h = tree.size().height();
        if !(w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0) {
            return None;
        }
        unsafe {
            if !width.is_null() {
                *width = w;
            }
            if !height.is_null() {
                *height = h;
            }
        }
        Some(1)
    });
    result.ok().flatten().unwrap_or(0)
}

/// Renders an SVG file into a heap-allocated straight RGBA8 buffer, scaled so
/// that the longer axis equals `max_dimension` (aspect ratio is preserved).
/// Caller receives the final pixel dimensions via `width`/`height` out-params.
///
/// Used for thumbnails and the first backdrop. Zoom does not go through this entry
/// point — see [`resvg_render_svg_region`].
///
/// Returns null on any failure (bad path, parse error, zero dimensions, panic).
/// On success, caller MUST free the buffer with `free_rust_buffer(ptr, width*height*4)`.
#[unsafe(no_mangle)]
pub extern "C" fn resvg_render_svg(
    path_ptr: *const c_char,
    max_dimension: c_int,
    width: *mut c_int,
    height: *mut c_int,
) -> *mut u8 {
    let result = catch_unwind(|| {
        if max_dimension <= 0 {
            return None;
        }
        let path = ptr_to_str(path_ptr)?;
        let tree = load_svg_tree(path)?;
        let svg_width = tree.size().width();
        let svg_height = tree.size().height();
        if !(svg_width > 0.0 && svg_height > 0.0) {
            return None;
        }

        // Fit the longer axis to max_dimension, preserving aspect ratio.
        let (w, h) = if svg_width >= svg_height {
            (max_dimension as f32, max_dimension as f32 * (svg_height / svg_width))
        } else {
            (max_dimension as f32 * (svg_width / svg_height), max_dimension as f32)
        };
        let (rw, rh) = (w.round() as u32, h.round() as u32);
        let pixels = render_svg_region(&tree, 0.0, 0.0, svg_width, svg_height, rw, rh)?;
        unsafe {
            safe_write_int(width, rw as c_int);
            safe_write_int(height, rh as c_int);
        }
        Some(Box::into_raw(pixels.into_boxed_slice()) as *mut u8)
    });
    result.ok().flatten().unwrap_or(std::ptr::null_mut())
}

/// Renders the SVG region `[src_x, src_y, src_w, src_h]` (user units) into an
/// `out_width`×`out_height` straight RGBA8 buffer.
///
/// This is how zoom stays sharp without a WebView: the view asks for the visible
/// slice at the current scale, so a 1-unit stroke stays one device pixel wide per
/// unit of zoom no matter how far the user goes. There is no document-size cap.
///
/// `out_width` and `out_height` must be in `1..=8192`. Returns null on failure.
/// On success, caller MUST free the buffer with `free_rust_buffer(ptr, width*height*4)`.
#[unsafe(no_mangle)]
pub extern "C" fn resvg_render_svg_region(
    path_ptr: *const c_char,
    src_x: f32,
    src_y: f32,
    src_w: f32,
    src_h: f32,
    out_width: c_int,
    out_height: c_int,
    width: *mut c_int,
    height: *mut c_int,
) -> *mut u8 {
    let result = catch_unwind(|| {
        if out_width <= 0 || out_height <= 0 {
            return None;
        }
        let path = ptr_to_str(path_ptr)?;
        let tree = load_svg_tree(path)?;
        let pixels = render_svg_region(
            &tree,
            src_x,
            src_y,
            src_w,
            src_h,
            out_width as u32,
            out_height as u32,
        )?;
        unsafe {
            safe_write_int(width, out_width);
            safe_write_int(height, out_height);
        }
        Some(Box::into_raw(pixels.into_boxed_slice()) as *mut u8)
    });
    result.ok().flatten().unwrap_or(std::ptr::null_mut())
}

/// Parses `path`, reusing a small cache so a zoom gesture does not re-read the file
/// and rebuild the font-resolved tree on every frame.
fn load_svg_tree(path: &str) -> Option<Arc<usvg::Tree>> {
    let meta = std::fs::metadata(path).ok()?;
    let len = meta.len();
    let modified_ms = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis())
        .unwrap_or(0);

    let mut cache = SVG_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(hit) = cache
        .iter()
        .find(|e| e.path == path && e.len == len && e.modified_ms == modified_ms)
    {
        return Some(hit.tree.clone());
    }

    let svg_data = std::fs::read(path).ok()?;
    let mut options = usvg::Options::default();
    options.fontdb = FONT_DB.clone();
    options.font_family = "Arial".to_string();
    let tree = Arc::new(usvg::Tree::from_data(&svg_data, &options).ok()?);

    cache.retain(|e| e.path != path);
    if cache.len() >= SVG_CACHE_LIMIT {
        cache.remove(0);
    }
    cache.push(CachedSvg {
        path: path.to_string(),
        len,
        modified_ms,
        tree: tree.clone(),
    });
    Some(tree)
}

/// Rasterises one axis-aligned region of `tree`.
///
/// Pixel `(0, 0)` of the output is SVG point `(src_x, src_y)`. The scale is
/// `out_w / src_w` by `out_h / src_h`, so the caller chooses the zoom by choosing
/// how many pixels a user-unit span occupies.
fn render_svg_region(
    tree: &usvg::Tree,
    src_x: f32,
    src_y: f32,
    src_w: f32,
    src_h: f32,
    out_w: u32,
    out_h: u32,
) -> Option<Vec<u8>> {
    if !(src_x.is_finite() && src_y.is_finite() && src_w.is_finite() && src_h.is_finite()) {
        return None;
    }
    if src_w <= 0.0 || src_h <= 0.0 || out_w == 0 || out_h == 0 {
        return None;
    }
    if out_w > SVG_MAX_EDGE || out_h > SVG_MAX_EDGE {
        return None;
    }
    // Reject a size that cannot be described as a byte length. The edge cap already
    // keeps this tiny; the check is the overflow guard for the allocation.
    let _bytes = (out_w as usize).checked_mul(out_h as usize)?.checked_mul(4)?;

    let mut pixmap = tiny_skia::Pixmap::new(out_w, out_h)?;
    // Translate the region origin to 0, then scale user units onto the pixmap.
    // pre_translate applies the translation before the scale.
    let transform = tiny_skia::Transform::from_scale(out_w as f32 / src_w, out_h as f32 / src_h)
        .pre_translate(-src_x, -src_y);
    resvg::render(tree, transform, &mut pixmap.as_mut());

    let mut pixels = pixmap.take();
    unpremultiply_straight(&mut pixels);
    Some(pixels)
}

/// tiny_skia writes premultiplied RGBA. FlyPhotos' Win2D bitmaps are straight
/// `R8G8B8A8UIntNormalized`, so convert in place. Fully transparent and fully
/// opaque pixels need no correction.
fn unpremultiply_straight(pixels: &mut [u8]) {
    pixels.par_chunks_exact_mut(4).for_each(|chunk| {
        let a = chunk[3];
        if a > 0 && a < 255 {
            let a_f = a as f32 / 255.0;
            chunk[0] = (chunk[0] as f32 / a_f).min(255.0) as u8;
            chunk[1] = (chunk[1] as f32 / a_f).min(255.0) as u8;
            chunk[2] = (chunk[2] as f32 / a_f).min(255.0) as u8;
        }
    });
}