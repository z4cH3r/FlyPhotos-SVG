#nullable enable
using System;
using System.Runtime.InteropServices;
using Windows.Graphics.DirectX;
using FlyPhotos.Core.Model;
using FlyPhotos.Infra.Interop;
using Microsoft.Graphics.Canvas;
using NLog;

namespace FlyPhotos.Display.ImageReading;

internal static unsafe partial class ResvgWrap
{
    private static readonly Logger Logger = LogManager.GetCurrentClassLogger();

    // Preview and the first backdrop only. Zoom re-renders the visible region at screen
    // resolution (see SvgVectorRenderer), so these sizes are not a quality cap.
    private const int PreviewMaxDimension = 800;
    private const int BackdropMaxDimension = 1024;
    /// <summary>Full-document raster used when the loaded DLL cannot render a zoomed region.</summary>
    private const int LegacyMaxDimension = 2000;

    /// <summary>True once the loaded fly_rust_bridge.dll is known to predate region rendering.</summary>
    private static bool _regionExportMissing;

    /// <summary>True after a region render finds no export. The viewer then keeps the backdrop.</summary>
    internal static bool RegionExportMissing => _regionExportMissing;

    public static (bool, PreviewDisplayItem) GetResized(ICanvasResourceCreatorWithDpi ctrl, string inputPath)
    {
        var (bmp, width, height) = RenderSvg(ctrl, inputPath, PreviewMaxDimension);
        if (bmp == null) return (false, PreviewDisplayItem.Empty());

        // Document size, not the preview bitmap. Fit and 100% zoom use these user units.
        // A DLL without the intrinsic-size export keeps the previous 2000 px metadata.
        if (!TryIntrinsicSize(inputPath, out double svgWidth, out double svgHeight))
        {
            int maxCurrent = Math.Max(width, height);
            double scale = maxCurrent > 0 ? (double)LegacyMaxDimension / maxCurrent : 1;
            svgWidth = width * scale;
            svgHeight = height * scale;
        }
        return (true, new PreviewDisplayItem(bmp, Origin.Disk, new ImageMetadata(svgWidth, svgHeight)));
    }

    public static (bool, HqDisplayItem) GetHq(ICanvasResourceCreatorWithDpi ctrl, string inputPath)
    {
        if (TryIntrinsicSize(inputPath, out double svgWidth, out double svgHeight))
        {
            var (bmp, _, _) = RenderSvg(ctrl, inputPath, BackdropMaxDimension);
            if (bmp == null) return (false, HqDisplayItem.Empty());
            return (true, new VectorHqDisplayItem(bmp, Origin.Disk, inputPath, svgWidth, svgHeight));
        }

        // Older fly_rust_bridge.dll: one full-document raster, same as before this change.
        var (legacy, _, _) = RenderSvg(ctrl, inputPath, LegacyMaxDimension);
        if (legacy == null) return (false, HqDisplayItem.Empty());
        return (true, new StaticHqDisplayItem(legacy, Origin.Disk));
    }

    /// <summary>
    /// Renders one region of an SVG into a straight RGBA buffer the caller owns.
    /// Returns null when the DLL has no region export (older builds) or the render fails.
    /// </summary>
    internal static byte[]? RenderRegion(string path, float srcX, float srcY, float srcW, float srcH,
        int outWidth, int outHeight)
    {
        if (_regionExportMissing || outWidth <= 0 || outHeight <= 0 || srcW <= 0 || srcH <= 0)
            return null;

        nint pixelPtr = nint.Zero;
        int width = 0, height = 0;
        try
        {
            pixelPtr = NativeRustBridge.resvg_render_svg_region(
                path, srcX, srcY, srcW, srcH, outWidth, outHeight, out width, out height);
            if (pixelPtr == nint.Zero || width <= 0 || height <= 0)
                return null;
            return CopyPixels(pixelPtr, width, height);
        }
        catch (EntryPointNotFoundException)
        {
            // fly_rust_bridge.dll next to the exe was built before region rendering.
            // Callers keep the backdrop raster from resvg_render_svg.
            _regionExportMissing = true;
            Logger.Warn("fly_rust_bridge.dll has no resvg_render_svg_region; SVG zoom keeps the backdrop raster");
            return null;
        }
        finally
        {
            if (pixelPtr != nint.Zero && width > 0 && height > 0)
            {
                long bytes = (long)width * height * 4;
                NativeRustBridge.free_rust_buffer(pixelPtr, (nuint)bytes);
            }
        }
    }

    private static bool TryIntrinsicSize(string path, out double width, out double height)
    {
        width = 0;
        height = 0;
        try
        {
            if (NativeRustBridge.resvg_svg_intrinsic_size(path, out float w, out float h) != 1)
                return false;
            if (w <= 0 || h <= 0 || float.IsNaN(w) || float.IsNaN(h))
                return false;
            width = w;
            height = h;
            return true;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
    }

    /// <summary>
    /// Calls the Rust resvg_render_svg entry point, copies the resulting RGBA8
    /// pixels into a <see cref="CanvasBitmap"/>, and unconditionally frees the
    /// native buffer — mirroring the contract in <see cref="RawlerWrapper"/>.
    /// </summary>
    private static (CanvasBitmap? Bitmap, int Width, int Height) RenderSvg(
        ICanvasResourceCreatorWithDpi ctrl, string inputPath, int maxDimension)
    {
        nint pixelPtr = NativeRustBridge.resvg_render_svg(
            inputPath, maxDimension, out int renderWidth, out int renderHeight);

        if (pixelPtr == nint.Zero || renderWidth <= 0 || renderHeight <= 0)
        {
            Logger.Warn("Rust failed to render SVG or returned empty dimensions: {InputPath}", inputPath);
            return (null, 0, 0);
        }

        try
        {
            byte[] pixels = CopyPixels(pixelPtr, renderWidth, renderHeight);
            var bitmap = CanvasBitmap.CreateFromBytes(
                ctrl, pixels, renderWidth, renderHeight, DirectXPixelFormat.R8G8B8A8UIntNormalized);
            return (bitmap, renderWidth, renderHeight);
        }
        catch (Exception ex)
        {
            Logger.Error(ex, "Failed to create CanvasBitmap from SVG render: {InputPath}", inputPath);
            return (null, 0, 0);
        }
        finally
        {
            long bytes = (long)renderWidth * renderHeight * 4;
            NativeRustBridge.free_rust_buffer(pixelPtr, (nuint)bytes);
        }
    }

    /// <summary>
    /// Copies a straight RGBA buffer out of the unmanaged allocation. The caller frees it.
    /// </summary>
    private static byte[] CopyPixels(nint ptr, int width, int height)
    {
        int totalBytes = checked(width * height * 4);
        byte[] pixels = GC.AllocateUninitializedArray<byte>(totalBytes);
        new ReadOnlySpan<byte>((void*)ptr, totalBytes).CopyTo(pixels);
        return pixels;
    }
}
