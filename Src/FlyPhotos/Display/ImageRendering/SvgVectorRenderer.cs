using System;
using System.Numerics;
using System.Threading.Tasks;
using Windows.Foundation;
using FlyPhotos.Core.Model;
using FlyPhotos.Display.ImageReading;
using FlyPhotos.Display.State;
using FlyPhotos.Infra.Configuration;
using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Brushes;
using Microsoft.Graphics.Canvas.UI.Xaml;
using NLog;
using Windows.Graphics.DirectX;

namespace FlyPhotos.Display.ImageRendering;

/// <summary>
/// Draws an SVG by re-rasterising the visible region at the current zoom.
///
/// ImageGlass 10 does the same thing with Svg.Skia: the picture stays vectors, and each frame
/// paints it through the view transform so only the screen is rasterised. Win2D has no SVG
/// picture, so resvg produces that screen-sized raster instead. Memory stays about one viewport;
/// zoom is not capped at a baked bitmap.
///
/// While a render is in flight the previous raster (and a low-resolution backdrop of the whole
/// document) keep the image on screen. The next settled frame swaps in the sharp one.
/// </summary>
internal partial class SvgVectorRenderer : IRenderer
{
    private static readonly Logger Logger = LogManager.GetCurrentClassLogger();

    /// <summary>One edge of a raster. A viewport is far smaller; this only rejects a runaway request.</summary>
    private const int MaxEdge = 8192;
    /// <summary>Extra SVG units around the viewport, in screen pixels, so a small pan stays inside the raster.</summary>
    private const double MarginPixels = 64;
    /// <summary>Re-render when zoom has moved by more than this fraction.</summary>
    private const double ScaleSlack = 0.02;

    private readonly CanvasAnimatedControl _canvas;
    private readonly string _path;
    private readonly double _svgWidth;
    private readonly double _svgHeight;
    private readonly bool _supportsTransparency;
    private readonly Action _invalidate;

    private CanvasImageBrush _checkeredBrush;
    public CanvasImageBrush CheckeredBrush { set => _checkeredBrush = value; }

    /// <summary>Full-document raster owned by the display item. Drawn under the sharp region.</summary>
    private readonly CanvasBitmap _backdrop;

    /// <summary>Latest viewport raster. Owned by this renderer.</summary>
    private CanvasBitmap _sharp;
    private Rect _sharpRect;
    private double _sharpScale;

    private readonly object _gate = new();
    private bool _renderInFlight;
    private bool _disposed;
    private Pending _pending;
    private double _failedScale = -1;
    private Rect _failedRect;

    private sealed class Pending(byte[] pixels, int width, int height, Rect rect, double scale)
    {
        public byte[] Pixels { get; } = pixels;
        public int Width { get; } = width;
        public int Height { get; } = height;
        public Rect Rect { get; } = rect;
        public double Scale { get; } = scale;
    }

    public SvgVectorRenderer(CanvasAnimatedControl canvas, VectorHqDisplayItem item,
        bool supportsTransparency, Action invalidate)
    {
        _canvas = canvas;
        _path = item.Path;
        _svgWidth = item.SvgWidth;
        _svgHeight = item.SvgHeight;
        _supportsTransparency = supportsTransparency;
        _invalidate = invalidate;
        _backdrop = item.Bitmap;
    }

    public void Draw(CanvasDrawingSession session, CanvasViewState viewState, CanvasImageInterpolation quality, bool isAnimating)
    {
        session.Units = CanvasUnits.Pixels;
        var drawCheckeredBackground = AppConfig.Settings.CheckeredBackground && _supportsTransparency;
        session.Antialiasing = drawCheckeredBackground ? CanvasAntialiasing.Aliased : CanvasAntialiasing.Antialiased;
        if (drawCheckeredBackground && _checkeredBrush != null)
        {
            _checkeredBrush.Transform = Matrix3x2.CreateScale(1f / viewState.Scale);
            session.FillRectangle(viewState.ImageRect, _checkeredBrush);
        }

        UploadPending(session);

        if (_backdrop != null)
            session.DrawImage(_backdrop, viewState.ImageRect, _backdrop.Bounds, 1f, quality);

        var scale = viewState.Scale;
        if (scale <= 0 || _svgWidth <= 0 || _svgHeight <= 0)
            return;

        var visible = VisibleImageRect(session, viewState);
        if (visible.IsEmpty)
            return;

        if (_sharp != null)
            session.DrawImage(_sharp, _sharpRect, _sharp.Bounds, 1f, quality);

        // Spring and touchpad bursts already redraw every tick. One render in flight is enough;
        // its completion wakes the canvas, which starts the next one if the view moved on.
        if (!Covers(visible, scale))
            RequestRender(visible, scale);
    }

    public void HandleScalingMethodChange()
    {
        DisposeSharp();
        _invalidate();
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _disposed = true;
            _pending = null;
        }
        DisposeSharp();
    }

    /// <summary>True when the sharp raster already covers <paramref name="visible"/> at <paramref name="scale"/>.</summary>
    private bool Covers(Rect visible, double scale)
    {
        if (_sharp == null || _sharpScale <= 0)
            return false;
        if (Math.Abs(_sharpScale - scale) > Math.Max(ScaleSlack * scale, 0.01))
            return false;
        return visible.X >= _sharpRect.X - 0.5
               && visible.Y >= _sharpRect.Y - 0.5
               && visible.Right <= _sharpRect.Right + 0.5
               && visible.Bottom <= _sharpRect.Bottom + 0.5;
    }

    private void RequestRender(Rect visible, double scale)
    {
        var margin = MarginPixels / scale;
        var src = Expand(visible, margin);
        if (src.Width <= 0 || src.Height <= 0)
            return;

        // Scale both edges together so a clamp cannot stretch the region. The stored
        // scale is the view scale we rendered for, so a clamped viewport does not
        // immediately request itself again.
        double fit = 1;
        double requestedW = src.Width * scale;
        double requestedH = src.Height * scale;
        if (requestedW > MaxEdge || requestedH > MaxEdge)
            fit = Math.Min(MaxEdge / requestedW, MaxEdge / requestedH);
        int outW = ClampEdge(requestedW * fit);
        int outH = ClampEdge(requestedH * fit);
        if (outW < 1 || outH < 1)
            return;
        // An older DLL has no region export. Stop scheduling renders; the backdrop stays on screen.
        if (ResvgWrap.RegionExportMissing)
            return;

        lock (_gate)
        {
            if (_disposed || _renderInFlight)
                return;
            if (SameFailure(src, scale))
                return;
            _renderInFlight = true;
        }

        _ = Task.Run(() =>
        {
            byte[] pixels = null;
            try
            {
                pixels = ResvgWrap.RenderRegion(_path, (float)src.X, (float)src.Y,
                    (float)src.Width, (float)src.Height, outW, outH);
            }
            catch (Exception ex)
            {
                Logger.Warn(ex, "SVG region render failed for {Path}", _path);
            }

            lock (_gate)
            {
                _renderInFlight = false;
                if (_disposed)
                    return;
                // outW and outH are capped at MaxEdge, so this product fits in an int.
                if (pixels == null || pixels.Length != outW * outH * 4)
                {
                    _failedScale = scale;
                    _failedRect = src;
                    return;
                }
                _failedScale = -1;
                _pending = new Pending(pixels, outW, outH, src, scale);
            }
            _invalidate();
        });
    }

    private bool SameFailure(Rect src, double scale)
    {
        if (_failedScale < 0)
            return false;
        return Math.Abs(_failedScale - scale) < 0.01
               && Math.Abs(_failedRect.X - src.X) < 0.5
               && Math.Abs(_failedRect.Y - src.Y) < 0.5
               && Math.Abs(_failedRect.Width - src.Width) < 0.5
               && Math.Abs(_failedRect.Height - src.Height) < 0.5;
    }

    private void UploadPending(CanvasDrawingSession session)
    {
        Pending pending;
        lock (_gate)
        {
            pending = _pending;
            _pending = null;
        }
        if (pending == null)
            return;

        try
        {
            var bitmap = CanvasBitmap.CreateFromBytes(session.Device, pending.Pixels,
                pending.Width, pending.Height, DirectXPixelFormat.R8G8B8A8UIntNormalized, 96f);
            DisposeSharp();
            _sharp = bitmap;
            _sharpRect = pending.Rect;
            _sharpScale = pending.Scale;
        }
        catch (Exception ex)
        {
            Logger.Warn(ex, "Failed to upload SVG raster for {Path}", _path);
        }
    }

    private Rect Expand(Rect visible, double margin)
    {
        var x0 = Math.Max(0, visible.X - margin);
        var y0 = Math.Max(0, visible.Y - margin);
        var x1 = Math.Min(_svgWidth, visible.Right + margin);
        var y1 = Math.Min(_svgHeight, visible.Bottom + margin);
        return new Rect(x0, y0, Math.Max(0, x1 - x0), Math.Max(0, y1 - y0));
    }

    private static int ClampEdge(double pixels)
    {
        if (double.IsNaN(pixels) || pixels <= 0)
            return 0;
        return (int)Math.Clamp(Math.Round(pixels), 1, MaxEdge);
    }

    /// <summary>The part of the SVG on screen, in user units.</summary>
    private Rect VisibleImageRect(CanvasDrawingSession session, CanvasViewState viewState)
    {
        var size = _canvas.Size;
        var dpi = session.Dpi / 96f;
        var w = (float)size.Width * dpi;
        var h = (float)size.Height * dpi;

        var p0 = Vector2.Transform(new Vector2(0, 0), viewState.MatInv);
        var p1 = Vector2.Transform(new Vector2(w, 0), viewState.MatInv);
        var p2 = Vector2.Transform(new Vector2(0, h), viewState.MatInv);
        var p3 = Vector2.Transform(new Vector2(w, h), viewState.MatInv);
        var min = Vector2.Min(Vector2.Min(p0, p1), Vector2.Min(p2, p3));
        var max = Vector2.Max(Vector2.Max(p0, p1), Vector2.Max(p2, p3));

        var rect = new Rect(new Point(min.X, min.Y), new Point(max.X, max.Y));
        rect.Intersect(viewState.ImageRect);
        return rect;
    }

    private void DisposeSharp()
    {
        _sharp?.Dispose();
        _sharp = null;
        _sharpScale = 0;
    }
}
