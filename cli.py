#!/usr/bin/env python3
"""
cli.py — Command-line interface for DocScanner.

Exposes all CV pipeline features from scanner.py with fine-grained control.
No Streamlit dependency.

Usage:
    python cli.py scan   -i photo.jpg -o scanned.png
    python cli.py detect -i photo.jpg
    python cli.py warp   -i photo.jpg --corners "100,50 800,60 810,1050 90,1040"
    python cli.py enhance -i photo.jpg --filter "Black & White"
    python cli.py batch  -i ./photos/ -o ./scanned/
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

from scanner import (
    FILTERS,
    apply_clahe_normalisation,
    downsample_for_display,
    draw_quad_on_image,
    find_document_quad,
    four_point_transform,
    fullimage_corners,
    order_points,
    resize_for_processing,
    run_pipeline,
    shadow_remove_for_segmentation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_image(path: str) -> np.ndarray:
    """Read an image file and return as BGR numpy array."""
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: could not decode image: {path}", file=sys.stderr)
        sys.exit(1)
    return img


def _read_image_bytes(path: str) -> bytes:
    """Read raw file bytes (for run_pipeline which expects bytes)."""
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "rb") as f:
        return f.read()


def _save_output(img: np.ndarray, output_path: str, fmt: str, quality: int,
                 quiet: bool) -> None:
    """Save image to disk in the requested format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if fmt == "pdf":
        try:
            import img2pdf
            import io
            # Encode to JPEG bytes first, then wrap in PDF
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            pdf_bytes = img2pdf.convert(io.BytesIO(buf.tobytes()))
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
        except ImportError:
            print("Error: img2pdf is required for PDF output. "
                  "Install with: pip install img2pdf", file=sys.stderr)
            sys.exit(1)
    elif fmt == "jpg":
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:  # png
        cv2.imwrite(output_path, img)

    if not quiet:
        h, w = img.shape[:2]
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  ✓ Saved {w}×{h} {fmt.upper()} → {output_path} ({size_kb:.1f} KB)")


def _save_debug_stages(stages: dict, output_path: str, quiet: bool) -> None:
    """Save all intermediate pipeline stages to a _debug/ folder."""
    base = os.path.splitext(output_path)[0]
    debug_dir = f"{base}_debug"
    os.makedirs(debug_dir, exist_ok=True)

    stage_names = [
        ("original", "01_original"),
        ("clahe", "02_clahe"),
        ("shadow_free", "03_shadow_free"),
        ("seg_mask", "04_seg_mask"),
        ("contour_vis", "05_contour_vis"),
        ("warped", "06_warped"),
        ("enhanced", "07_enhanced"),
    ]

    saved = 0
    for key, filename in stage_names:
        if key not in stages:
            continue
        img = stages[key]
        if img is None:
            continue
        if img.ndim == 2:  # grayscale mask
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        path = os.path.join(debug_dir, f"{filename}.png")
        cv2.imwrite(path, img)
        saved += 1

    if not quiet:
        print(f"  ✓ Debug: saved {saved} stage images → {debug_dir}/")


def _parse_corners(corners_str: str) -> np.ndarray:
    """
    Parse corner string like "100,50 800,60 810,1050 90,1040"
    into a (4,2) float32 array.
    """
    parts = corners_str.strip().split()
    if len(parts) != 4:
        print(f"Error: expected 4 corner points, got {len(parts)}. "
              f"Format: x1,y1 x2,y2 x3,y3 x4,y4", file=sys.stderr)
        sys.exit(1)
    try:
        pts = []
        for p in parts:
            x, y = p.split(",")
            pts.append([float(x), float(y)])
        return np.array(pts, dtype=np.float32)
    except (ValueError, IndexError):
        print("Error: corners must be in format: x1,y1 x2,y2 x3,y3 x4,y4",
              file=sys.stderr)
        sys.exit(1)


def _auto_output_path(input_path: str, suffix: str, fmt: str) -> str:
    """Generate a default output path from the input path."""
    base, _ = os.path.splitext(input_path)
    ext = f".{fmt}" if fmt != "pdf" else ".pdf"
    return f"{base}_{suffix}{ext}"


def _print_timings(stages: dict) -> None:
    """Print pipeline timing information."""
    timings = stages.get("timings", {})
    total = stages.get("total_time", 0.0)
    method = stages.get("corner_method", "—")
    edge = stages.get("edge_method", "—")

    label_map = {
        "decode":        "Decode        ",
        "resize":        "Resize (1080) ",
        "clahe":         "CLAHE         ",
        "shadow_remove": "Shadow removal",
        "detection":     "Detection     ",
        "warp":          "Perspective   ",
        "enhance":       "Enhancement   ",
    }

    print("\n  ⏱  Stage Timings")
    print("  " + "─" * 36)
    for key, label in label_map.items():
        ms = timings.get(key, 0.0)
        bar = "█" * int(ms / 10) if ms > 0 else ""
        print(f"  {label}  {ms:7.1f} ms  {bar}")
    print("  " + "─" * 36)

    speed = "✓ under 1s" if total < 1000 else "✗ over 1s"
    print(f"  Total: {total:.0f} ms — {speed}")
    print(f"  Method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_scan(args: argparse.Namespace) -> None:
    """Full auto pipeline: detect → warp → enhance → save."""
    image_bytes = _read_image_bytes(args.input)

    if not args.quiet:
        print(f"  Scanning: {args.input}")

    stages = run_pipeline(image_bytes, filter_name=args.filter)

    if "error" in stages:
        print(f"Error: {stages['error']}", file=sys.stderr)
        sys.exit(1)

    output = args.output or _auto_output_path(args.input, "scanned", args.format)
    _save_output(stages["enhanced"], output, args.format, args.quality, args.quiet)

    if args.verbose:
        _print_timings(stages)

    if args.debug:
        _save_debug_stages(stages, output, args.quiet)


def cmd_detect(args: argparse.Namespace) -> None:
    """Detect document corners and output coordinates."""
    image_bytes = _read_image_bytes(args.input)

    stages = run_pipeline(image_bytes, filter_name="Original")
    if "error" in stages:
        print(f"Error: {stages['error']}", file=sys.stderr)
        sys.exit(1)

    corners = stages["corners_proc"]  # (4,2) in processing space
    scale = stages["scale"]
    method = stages.get("corner_method", "unknown")

    # Convert to original image coordinates
    corners_orig = corners / scale

    if args.json:
        result = {
            "corners": corners_orig.tolist(),
            "method": method,
            "labels": ["TL", "TR", "BR", "BL"],
            "coordinate_space": "original_image",
        }
        if args.verbose:
            result["timings"] = stages.get("timings", {})
            result["total_time_ms"] = stages.get("total_time", 0.0)
        print(json.dumps(result, indent=2))
    else:
        labels = ["TL", "TR", "BR", "BL"]
        if not args.quiet:
            print(f"  Method: {method}")
            print(f"  Corners (original image coordinates):")
        for label, pt in zip(labels, corners_orig):
            print(f"    {label}: ({pt[0]:.1f}, {pt[1]:.1f})")

    if args.verbose and not args.json:
        _print_timings(stages)

    # Optionally save a visualisation
    if args.output:
        vis = draw_quad_on_image(stages["resized"].copy(), corners)
        _save_output(vis, args.output, args.format, args.quality, args.quiet)


def cmd_warp(args: argparse.Namespace) -> None:
    """Warp with user-supplied corners."""
    corners = _parse_corners(args.corners)
    image_bytes = _read_image_bytes(args.input)

    if not args.quiet:
        print(f"  Warping: {args.input}")
        labels = ["TL", "TR", "BR", "BL"]
        ordered = order_points(corners)
        for label, pt in zip(labels, ordered):
            print(f"    {label}: ({pt[0]:.1f}, {pt[1]:.1f})")

    # Run pipeline with manual corners (corners are in original image space)
    original = _read_image(args.input)
    oh, ow = original.shape[:2]
    resized, scale = resize_for_processing(original, max_edge=args.max_edge)

    # Scale user corners to processing space
    corners_proc = corners * scale
    stages = run_pipeline(image_bytes, filter_name=args.filter,
                          manual_corners=corners_proc)

    if "error" in stages:
        print(f"Error: {stages['error']}", file=sys.stderr)
        sys.exit(1)

    output = args.output or _auto_output_path(args.input, "warped", args.format)
    _save_output(stages["enhanced"], output, args.format, args.quality, args.quiet)

    if args.verbose:
        _print_timings(stages)

    if args.debug:
        _save_debug_stages(stages, output, args.quiet)


def cmd_enhance(args: argparse.Namespace) -> None:
    """Apply a filter to any image without corner detection or warping."""
    img = _read_image(args.input)

    if not args.quiet:
        print(f"  Enhancing: {args.input} with '{args.filter}' filter")

    filter_fn = FILTERS.get(args.filter)
    if filter_fn is None:
        print(f"Error: unknown filter '{args.filter}'. "
              f"Choose from: {', '.join(FILTERS.keys())}", file=sys.stderr)
        sys.exit(1)

    t = time.perf_counter()
    result = filter_fn(img)
    elapsed = (time.perf_counter() - t) * 1000

    output = args.output or _auto_output_path(args.input, "enhanced", args.format)
    _save_output(result, output, args.format, args.quality, args.quiet)

    if args.verbose:
        print(f"\n  ⏱  Enhancement: {elapsed:.1f} ms")


def cmd_batch(args: argparse.Namespace) -> None:
    """Process a directory of images through the full pipeline."""
    input_dir = args.input
    if not os.path.isdir(input_dir):
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect image files
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    files = sorted(set(files))

    if not files:
        print(f"Error: no image files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or os.path.join(input_dir, "scanned")
    os.makedirs(output_dir, exist_ok=True)

    if not args.quiet:
        print(f"  Batch processing {len(files)} images")
        print(f"  Output directory: {output_dir}")
        print(f"  Filter: {args.filter}")
        print()

    success = 0
    failed = 0
    total_time = 0.0

    for i, filepath in enumerate(files, 1):
        basename = os.path.basename(filepath)
        name, _ = os.path.splitext(basename)
        ext = f".{args.format}" if args.format != "pdf" else ".pdf"
        out_path = os.path.join(output_dir, f"{name}_scanned{ext}")

        if not args.quiet:
            print(f"  [{i}/{len(files)}] {basename}...", end=" ", flush=True)

        try:
            image_bytes = _read_image_bytes(filepath)
            stages = run_pipeline(image_bytes, filter_name=args.filter)

            if "error" in stages:
                if not args.quiet:
                    print(f"✗ {stages['error']}")
                failed += 1
                continue

            _save_output(stages["enhanced"], out_path, args.format,
                         args.quality, quiet=True)
            elapsed = stages.get("total_time", 0.0)
            total_time += elapsed

            method = stages.get("corner_method", "?")
            if not args.quiet:
                print(f"✓ ({elapsed:.0f} ms, {method})")

            if args.debug:
                _save_debug_stages(stages, out_path, quiet=True)

            success += 1

        except Exception as e:
            if not args.quiet:
                print(f"✗ {e}")
            failed += 1

    if not args.quiet:
        print(f"\n  ─────────────────────────────────")
        print(f"  Done: {success} succeeded, {failed} failed")
        print(f"  Total processing time: {total_time:.0f} ms")
        print(f"  Output: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docscanner",
        description="DocScanner CLI — Document scanning with fine-grained CV control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py scan -i photo.jpg                          Auto-detect & scan
  python cli.py scan -i photo.jpg --filter "Black & White" Scan with B&W filter
  python cli.py scan -i photo.jpg --debug --verbose        Full diagnostics
  python cli.py detect -i photo.jpg --json                 Get corners as JSON
  python cli.py warp -i photo.jpg --corners "10,10 800,10 800,1000 10,1000"
  python cli.py enhance -i photo.jpg --filter "Magic Colour"
  python cli.py batch -i ./photos/ -o ./scanned/           Batch process folder
""",
    )

    # Global flags (also added to each subcommand for flexible placement)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed timing and method information")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress all output except errors")

    subparsers = parser.add_subparsers(dest="command", required=True,
                                        help="Available commands")

    # ── Common arguments factory ──────────────────────────────────────────
    def add_common(p: argparse.ArgumentParser, output_required: bool = False,
                   add_filter: bool = True) -> None:
        p.add_argument("-i", "--input", required=True,
                       help="Input image path (or directory for batch)")
        p.add_argument("-o", "--output", required=output_required, default=None,
                       help="Output path (auto-generated if omitted)")
        if add_filter:
            p.add_argument("--filter", default="Original",
                           choices=list(FILTERS.keys()),
                           help="Enhancement filter (default: Original)")
        p.add_argument("--format", default="png", choices=["png", "jpg", "pdf"],
                       help="Output format (default: png)")
        p.add_argument("--quality", type=int, default=95,
                       help="JPEG/PDF quality 1-100 (default: 95)")
        p.add_argument("--debug", action="store_true",
                       help="Save intermediate pipeline stages")
        p.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed timing and method information")
        p.add_argument("-q", "--quiet", action="store_true",
                       help="Suppress all output except errors")

    # ── scan ──────────────────────────────────────────────────────────────
    p_scan = subparsers.add_parser("scan",
        help="Full auto pipeline: detect corners → warp → enhance → save",
        description="Automatically detect document corners, apply perspective "
                    "correction, and enhance with a filter.")
    add_common(p_scan)
    p_scan.add_argument("--max-edge", type=int, default=1080,
                        help="Max processing resolution in px (default: 1080)")
    p_scan.set_defaults(func=cmd_scan)

    # ── detect ────────────────────────────────────────────────────────────
    p_detect = subparsers.add_parser("detect",
        help="Detect document corners and output coordinates",
        description="Run corner detection and print the 4 corner coordinates. "
                    "Optionally save a visualisation image.")
    p_detect.add_argument("-i", "--input", required=True,
                          help="Input image path")
    p_detect.add_argument("-o", "--output", default=None,
                          help="Save corner visualisation to this path")
    p_detect.add_argument("--json", action="store_true",
                          help="Output corners as JSON")
    p_detect.add_argument("--format", default="png", choices=["png", "jpg"],
                          help="Visualisation format (default: png)")
    p_detect.add_argument("--quality", type=int, default=95,
                          help="JPEG quality 1-100 (default: 95)")
    p_detect.add_argument("--debug", action="store_true",
                          help="Save intermediate pipeline stages")
    p_detect.add_argument("-v", "--verbose", action="store_true",
                          help="Show detailed timing and method information")
    p_detect.add_argument("-q", "--quiet", action="store_true",
                          help="Suppress all output except errors")
    p_detect.set_defaults(func=cmd_detect)

    # ── warp ──────────────────────────────────────────────────────────────
    p_warp = subparsers.add_parser("warp",
        help="Perspective warp with user-supplied corners",
        description="Apply perspective correction using manually specified "
                    "corner coordinates (in original image pixel space).")
    add_common(p_warp)
    p_warp.add_argument("--corners", required=True,
                        help='4 corner points: "x1,y1 x2,y2 x3,y3 x4,y4" '
                             '(TL TR BR BL in original image pixels)')
    p_warp.add_argument("--max-edge", type=int, default=1080,
                        help="Max processing resolution in px (default: 1080)")
    p_warp.set_defaults(func=cmd_warp)

    # ── enhance ───────────────────────────────────────────────────────────
    p_enhance = subparsers.add_parser("enhance",
        help="Apply a filter to any image (no warp)",
        description="Apply one of the enhancement filters to an image "
                    "without performing corner detection or perspective warp.")
    p_enhance.add_argument("-i", "--input", required=True,
                           help="Input image path")
    p_enhance.add_argument("-o", "--output", default=None,
                           help="Output path (auto-generated if omitted)")
    p_enhance.add_argument("--filter", default="Original",
                           choices=list(FILTERS.keys()),
                           help="Enhancement filter (default: Original)")
    p_enhance.add_argument("--format", default="png", choices=["png", "jpg", "pdf"],
                           help="Output format (default: png)")
    p_enhance.add_argument("--quality", type=int, default=95,
                           help="JPEG/PDF quality 1-100 (default: 95)")
    p_enhance.add_argument("--debug", action="store_true",
                           help="(no effect for enhance)")
    p_enhance.add_argument("-v", "--verbose", action="store_true",
                           help="Show detailed timing and method information")
    p_enhance.add_argument("-q", "--quiet", action="store_true",
                           help="Suppress all output except errors")
    p_enhance.set_defaults(func=cmd_enhance)

    # ── batch ─────────────────────────────────────────────────────────────
    p_batch = subparsers.add_parser("batch",
        help="Batch process a directory of images",
        description="Process all images in a directory through the full "
                    "scan pipeline (detect → warp → enhance).")
    add_common(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
