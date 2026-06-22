"""
Visual verification: Place each tile on a composite-sized canvas at its matched position.

Usage:
    python verify_positions.py <positions.json> <composite.png> <tiles_dir> <output_dir> [--crop-bottom 60] [--min-confidence 0.0]

Opens the output_dir in gwenview (or any viewer) and flip through images.
The first image (000_composite.png) is the composite itself for reference.

Each subsequent image shows ONE tile placed at its detected position on a
gray canvas the same size as the composite. The filename encodes the confidence.
"""

import cv2
import numpy as np
import os
import sys
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate per-tile verification images")
    parser.add_argument("positions_json", help="Path to positions.json from find_tile_positions.py")
    parser.add_argument("composite", help="Path to composite PNG image")
    parser.add_argument("tiles_dir", help="Directory containing tile images")
    parser.add_argument("output_dir", help="Output directory for verification images")
    parser.add_argument("--crop-bottom", type=int, default=60, help="Pixels to crop from bottom of tiles (default: 60)")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Skip tiles below this confidence (default: 0.0 = all)")
    parser.add_argument("--blend", type=float, default=0.0,
                        help="If >0, blend composite underneath at this alpha (0.0-1.0). "
                             "0.0 = dark background only, 0.3 = faint composite behind tile")
    args = parser.parse_args()

    # Load positions
    with open(args.positions_json) as f:
        positions = json.load(f)

    # Load composite
    comp = cv2.imread(args.composite, cv2.IMREAD_UNCHANGED)
    if comp is None:
        print(f"Cannot load composite: {args.composite}")
        sys.exit(1)

    # Get composite dimensions (use BGR only)
    if comp.ndim == 3 and comp.shape[2] == 4:
        comp_bgr = comp[:, :, :3]
    elif comp.ndim == 3:
        comp_bgr = comp
    else:
        comp_bgr = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)

    h, w = comp_bgr.shape[:2]
    print(f"Composite size: {w}x{h}")
    print(f"Tiles: {len(positions)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save composite as first image for reference
    ref_path = os.path.join(args.output_dir, "000_composite.png")
    cv2.imwrite(ref_path, comp_bgr)
    print(f"Saved reference: {ref_path}")

    # Prepare dim composite background if blending
    if args.blend > 0:
        bg_base = (comp_bgr * args.blend).astype(np.uint8)
    else:
        bg_base = None

    # Sort by filename for consistent ordering
    sorted_tiles = sorted(positions.items())
    total = len(sorted_tiles)
    skipped = 0

    for idx, (fname, info) in enumerate(sorted_tiles):
        conf = info["confidence"]
        if conf < args.min_confidence:
            skipped += 1
            continue

        x, y = info["x"], info["y"]
        tw, th = info["w"], info["h"]

        # Load tile
        tile_path = os.path.join(args.tiles_dir, fname)
        tile = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED)
        if tile is None:
            print(f"  [SKIP] Cannot load: {fname}")
            skipped += 1
            continue

        # Crop bottom
        if args.crop_bottom > 0 and tile.shape[0] > args.crop_bottom:
            tile = tile[:tile.shape[0] - args.crop_bottom, :]

        # Convert to BGR
        if tile.ndim == 3 and tile.shape[2] == 4:
            tile_bgr = tile[:, :, :3]
        elif tile.ndim == 3:
            tile_bgr = tile
        else:
            tile_bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

        # Create canvas
        if bg_base is not None:
            canvas = bg_base.copy()
        else:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Place tile (handle boundary clipping)
        tile_h, tile_w = tile_bgr.shape[:2]
        # Source region
        sx1, sy1 = 0, 0
        sx2, sy2 = tile_w, tile_h
        # Dest region
        dx1, dy1 = x, y
        dx2, dy2 = x + tile_w, y + tile_h

        # Clip to canvas bounds
        if dx1 < 0:
            sx1 -= dx1
            dx1 = 0
        if dy1 < 0:
            sy1 -= dy1
            dy1 = 0
        if dx2 > w:
            sx2 -= (dx2 - w)
            dx2 = w
        if dy2 > h:
            sy2 -= (dy2 - h)
            dy2 = h

        if sx1 < sx2 and sy1 < sy2:
            canvas[dy1:dy2, dx1:dx2] = tile_bgr[sy1:sy2, sx1:sx2]

        # Draw thin border around tile position
        cv2.rectangle(canvas, (dx1, dy1), (dx2 - 1, dy2 - 1), (0, 255, 0) if conf >= 0.8 else (0, 0, 255), 1)

        # Status tag
        status = "OK" if conf >= 0.8 else "LOW"
        out_name = f"{idx + 1:04d}_{status}_{conf:.4f}_{os.path.splitext(fname)[0]}.png"
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, canvas)

        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [{idx + 1}/{total}] {out_name}")

    print(f"\nDone! {total - skipped} images saved to {args.output_dir}/")
    if skipped:
        print(f"  ({skipped} skipped)")
    print(f"\nTip: open in gwenview and flip through with arrow keys.")
    print(f"  First image is the composite for reference.")


if __name__ == "__main__":
    main()