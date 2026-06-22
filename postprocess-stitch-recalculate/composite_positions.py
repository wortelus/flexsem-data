"""
Template Matching: Find tile positions in a grayscale composite image.

Usage:
    python find_tile_positions.py <composite.png> <tiles_dir> [--crop-bottom 60] [--threshold 0.8]

- composite.png: The stitched composite (PNG with alpha)
- tiles_dir: Directory with individual tile images
- --crop-bottom: Pixels to crop from bottom of each tile (default: 60)
- --threshold: Match threshold (default: 0.8)

Output:
- Prints a dict of {filename: (x, y)} positions
- Saves a verification image showing colored rectangles on the composite
"""

import cv2
import numpy as np
import os
import sys
import argparse
import json


def load_composite_gray(path):
    """Load composite PNG, convert to grayscale ignoring alpha."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load composite: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        alpha = img[:, :, 3]
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alpha = np.full(gray.shape, 255, dtype=np.uint8)
    else:
        gray = img
        alpha = np.full(gray.shape, 255, dtype=np.uint8)

    # median filter
    # gray = cv2.medianBlur(gray, 5)
    # gaussian filter
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray, alpha, img


def load_tile_gray(path, crop_bottom=60):
    """Load a tile image, crop bottom pixels, convert to grayscale."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None

    if 0 < crop_bottom < img.shape[0]:
        img = img[: img.shape[0] - crop_bottom, :]

    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # median filter
    # gray = cv2.medianBlur(gray, 5)
    # gaussian filter
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray, img


def find_tile_position(composite_gray, composite_alpha, tile_gray):
    th, tw = tile_gray.shape[:2]
    ch, cw = composite_gray.shape[:2]
    if th > ch or tw > cw:
        return None, -1.0

    # Nahradíme průhledné pixely neutrální hodnotou (průměr obrazu),
    # aby neovlivnily korelaci – nebudou ani zlepšovat ani zhoršovat shodu
    mask = composite_alpha < 128
    img = composite_gray.copy()
    img[mask] = np.mean(composite_gray[~mask]).astype(np.uint8)

    result = cv2.matchTemplate(img, tile_gray, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val


def generate_colors(n):
    """Generate n distinct colors for visualization."""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def main():
    parser = argparse.ArgumentParser(description="Find tile positions in composite via template matching")
    parser.add_argument("composite", help="Path to composite PNG image")
    parser.add_argument("tiles_dir", help="Directory containing tile images")
    parser.add_argument("--crop-bottom", type=int, default=60, help="Pixels to crop from bottom of tiles (default: 60)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Match confidence threshold (default: 0.8)")
    parser.add_argument("--output", default=None, help="Output verification image path (default: composite_verified.png)")
    parser.add_argument("--output-json", default=None, help="Output JSON with positions (default: positions.json)")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.composite))[0]
        args.output = f"{base}_verified.png"
    if args.output_json is None:
        args.output_json = "positions.json"

    # Load composite
    print(f"Loading composite: {args.composite}")
    comp_gray, comp_alpha, comp_raw = load_composite_gray(args.composite)
    print(f"  Composite size: {comp_gray.shape[1]}x{comp_gray.shape[0]}")

    # Prepare visualization (BGR)
    if comp_raw.ndim == 2:
        vis = cv2.cvtColor(comp_raw, cv2.COLOR_GRAY2BGR)
    elif comp_raw.shape[2] == 4:
        vis = comp_raw[:, :, :3].copy()
    else:
        vis = comp_raw.copy()

    # Find tile files
    tile_extensions = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
    tile_files = sorted([
        f for f in os.listdir(args.tiles_dir)
        if os.path.splitext(f)[1].lower() in tile_extensions
    ])

    if not tile_files:
        print(f"No tile images found in {args.tiles_dir}")
        sys.exit(1)

    print(f"Found {len(tile_files)} tile(s) in {args.tiles_dir}")
    print(f"Cropping bottom {args.crop_bottom}px from each tile\n")

    colors = generate_colors(len(tile_files))
    positions = {}
    results = []

    for i, fname in enumerate(tile_files):
        fpath = os.path.join(args.tiles_dir, fname)
        tile_gray, tile_raw = load_tile_gray(fpath, args.crop_bottom)
        if tile_gray is None:
            print(f"  [SKIP] {fname}: could not load")
            continue

        loc, confidence = find_tile_position(comp_gray, comp_alpha, tile_gray)

        if loc is None:
            print(f"  [SKIP] {fname}: tile larger than composite")
            continue

        th, tw = tile_gray.shape[:2]
        status = "OK" if confidence >= args.threshold else "LOW"
        print(f"  [{status}] {fname}: pos=({loc[0]}, {loc[1]}), size=({tw}x{th}), confidence={confidence:.4f}")

        positions[fname] = {"x": loc[0], "y": loc[1], "w": tw, "h": th, "confidence": round(float(confidence), 4)}
        results.append((fname, loc, tw, th, confidence, colors[i]))

    # Draw rectangles on verification image
    print(f"\nGenerating verification image...")
    for fname, loc, tw, th, conf, color in results:
        x, y = loc
        thickness = 3 if conf >= args.threshold else 1
        cv2.rectangle(vis, (x, y), (x + tw, y + th), color, thickness)
        label = f"{fname} ({conf:.2f})"
        font_scale = 0.4
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x, y - lh - 6), (x + lw + 4, y), color, -1)
        cv2.putText(vis, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(args.output, vis)
    print(f"  Saved: {args.output}")

    # Save JSON
    with open(args.output_json, "w") as f:
        json.dump(positions, f, indent=2)
    print(f"  Saved: {args.output_json}")

    # Print dict for quick copy-paste
    print(f"\n# Positions dict:")
    print("positions = {")
    for fname, info in positions.items():
        print(f'    "{fname}": ({info["x"]}, {info["y"]}),  # {info["w"]}x{info["h"]}, conf={info["confidence"]}')
    print("}")


if __name__ == "__main__":
    main()