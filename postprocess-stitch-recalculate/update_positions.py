"""
Update x_actual_abs / y_actual_abs in experiment JSON using pixel positions
from template matching (positions.json).

Usage:
    python update_positions.py <experiment.json> <positions.json> [--pixel-size 12.40234] [--output updated.json]

Logic:
    - Reference tile (step 0) has known absolute position in nm AND a pixel position.
    - For every other tile:
        x_actual_abs = ref_x_nm + (tile_px_x - ref_px_x) * pixel_size
        y_actual_abs = ref_y_nm - (tile_px_y - ref_px_y) * pixel_size
      (Y is inverted because image Y grows downward, stage Y grows upward)

Notes:
    - Filenames are matched between experiment img_path (basename) and positions.json keys.
    - If --invert-y is NOT set, stage Y is assumed to grow upward (image Y inverted).
"""

import json
import argparse
import os
import sys


def basename_from_path(p):
    """Extract bare filename from img_path like 'temp/foo.bmp'."""
    return os.path.basename(p)


def main():
    parser = argparse.ArgumentParser(description="Update actual positions from template matching pixels")
    parser.add_argument("experiment", help="Path to experiment JSON")
    parser.add_argument("positions", help="Path to positions.json from template matching")
    parser.add_argument("--pixel-size", type=float, default=12.40234,
                        help="Pixel size in nm (default: 12.40234)")
    parser.add_argument("--ref-step", type=int, default=0,
                        help="Step number to use as reference (default: 0)")
    parser.add_argument("--no-invert-y", action="store_true",
                        help="Don't invert Y axis (by default Y is inverted: image down = stage up)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: experiment_updated.json)")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.experiment)
        args.output = f"{base}_updated{ext}"

    x_sign = -1.0
    y_sign = 1.0

    # Load data (support both .json and .jsonl)
    with open(args.experiment) as f:
        content = f.read().strip()

    try:
        # Try loading as a single standard JSON array/object
        entries = json.loads(content)
    except json.JSONDecodeError:
        # If that fails, treat it as JSONL (one JSON object per line)
        print("Detected JSONL format, parsing line by line...")
        entries = [json.loads(line) for line in content.splitlines() if line.strip()]

    with open(args.positions) as f:
        pixel_positions = json.load(f)

    print(f"Loaded {len(entries)} experiment entries")
    print(f"Loaded {len(pixel_positions)} pixel positions")
    print(f"Pixel size: {args.pixel_size} nm/px")
    print(f"Y axis: {'same as image' if args.no_invert_y else 'inverted (image down = stage up)'}")
    print()

    # Build lookup: basename -> pixel position
    px_lookup = {}
    for fname, info in pixel_positions.items():
        px_lookup[fname] = (info["x"], info["y"])

    # Find reference entry
    ref_entry = None
    for e in entries:
        if e["step"] == args.ref_step:
            ref_entry = e
            break

    if ref_entry is None:
        print(f"ERROR: No entry with step={args.ref_step} found!")
        sys.exit(1)

    ref_basename = basename_from_path(ref_entry["img_path"])
    if ref_basename not in px_lookup:
        print(f"ERROR: Reference file '{ref_basename}' not found in positions.json!")
        print(f"  Available keys: {list(pixel_positions.keys())[:5]}...")
        sys.exit(1)

    ref_px_x, ref_px_y = px_lookup[ref_basename]
    ref_nm_x = ref_entry["x_target_abs"]  # reference actual = target (known good)
    ref_nm_y = ref_entry["y_target_abs"]

    print(f"Reference (step {args.ref_step}):")
    print(f"  File: {ref_basename}")
    print(f"  Pixel pos: ({ref_px_x}, {ref_px_y})")
    print(f"  Abs pos (nm): ({ref_nm_x}, {ref_nm_y})")
    print()

    # Update all entries
    updated = 0
    skipped = 0
    for e in entries:
        bname = basename_from_path(e["img_path"])

        if bname not in px_lookup:
            print(f"  [SKIP] step {e['step']}: '{bname}' not in positions.json")
            skipped += 1
            continue

        px_x, px_y = px_lookup[bname]

        dx_px = px_x - ref_px_x
        dy_px = px_y - ref_px_y

        old_x = e["x_actual_abs"]
        old_y = e["y_actual_abs"]

        new_x = round(ref_nm_x + x_sign * dx_px * args.pixel_size)
        new_y = round(ref_nm_y + y_sign * dy_px * args.pixel_size)

        e["x_actual_abs"] = new_x
        e["y_actual_abs"] = new_y

        dx_nm = new_x - ref_nm_x
        dy_nm = new_y - ref_nm_y

        print(f"  [step {e['step']:>3d}] px=({px_x:>5d}, {px_y:>5d})  "
              f"dx_px=({dx_px:>+5d}, {dy_px:>+5d})  "
              f"old=({old_x:>7d}, {old_y:>7d}) -> new=({new_x:>7d}, {new_y:>7d})  "
              f"delta_nm=({dx_nm:>+8.0f}, {dy_nm:>+8.0f})")
        updated += 1

    print(f"\nUpdated: {updated}, Skipped: {skipped}")

    # Sanity check: correlation between target and actual deltas
    print(f"\n{'='*60}")
    print("SANITY CHECK: target delta vs. actual delta direction")
    print(f"{'='*60}")
    dx_targets = []
    dy_targets = []
    dx_actuals = []
    dy_actuals = []
    for e in entries:
        if e["step"] == args.ref_step:
            continue
        dt_x = e["x_target_abs"] - ref_nm_x
        dt_y = e["y_target_abs"] - ref_nm_y
        da_x = e["x_actual_abs"] - ref_nm_x
        da_y = e["y_actual_abs"] - ref_nm_y
        dx_targets.append(dt_x)
        dy_targets.append(dt_y)
        dx_actuals.append(da_x)
        dy_actuals.append(da_y)

    if len(dx_targets) > 1:
        # Sign agreement: how often do target and actual deltas point the same direction?
        x_agree = sum(1 for dt, da in zip(dx_targets, dx_actuals) if dt * da > 0)
        y_agree = sum(1 for dt, da in zip(dy_targets, dy_actuals) if dt * da > 0)
        n = len(dx_targets)

        print(f"  X direction agreement: {x_agree}/{n} ({100*x_agree/n:.0f}%)")
        print(f"  Y direction agreement: {y_agree}/{n} ({100*y_agree/n:.0f}%)")

        # Pearson correlation
        def pearson(a, b):
            a = [float(v) for v in a]
            b = [float(v) for v in b]
            n = len(a)
            ma = sum(a) / n
            mb = sum(b) / n
            num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
            den_a = sum((ai - ma) ** 2 for ai in a) ** 0.5
            den_b = sum((bi - mb) ** 2 for bi in b) ** 0.5
            if den_a == 0 or den_b == 0:
                return 0.0
            return num / (den_a * den_b)

        r_x = pearson(dx_targets, dx_actuals)
        r_y = pearson(dy_targets, dy_actuals)
        print(f"  X Pearson correlation:  {r_x:+.4f}")
        print(f"  Y Pearson correlation:  {r_y:+.4f}")

        # Mean absolute error (hysteresis)
        mae_x = sum(abs(dt - da) for dt, da in zip(dx_targets, dx_actuals)) / n
        mae_y = sum(abs(dt - da) for dt, da in zip(dy_targets, dy_actuals)) / n
        print(f"  X mean abs error (nm):  {mae_x:.0f}")
        print(f"  Y mean abs error (nm):  {mae_y:.0f}")

        if r_x < 0.3 or r_y < 0.3:
            print(f"\n  ⚠ WARNING: Low correlation detected!")
            print(f"    This may indicate inverted axis convention.")
            print(f"    Try running with {'--no-invert-y' if not args.no_invert_y else 'without --no-invert-y'}.")
        elif r_x > 0.8 and r_y > 0.8:
            print(f"\n  ✓ Looks good! Target and actual deltas are well correlated.")
        else:
            print(f"\n  ~ Moderate correlation. Check results visually.")
    else:
        print("  Not enough data points for sanity check.")

    # Save (preserve format)
    with open(args.output, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()