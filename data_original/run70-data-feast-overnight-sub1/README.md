`python ..\..\postprocess-stitch-recalculate\update_positions.py .\hysteresis_dataset_reconstructed.jsonl .\positions.json --ref-step 1640    `

Updated: 2737, Skipped: 0

============================================================
SANITY CHECK: target delta vs. actual delta direction
============================================================
  X direction agreement: 2401/2736 (88%)
  Y direction agreement: 2519/2736 (92%)
  X Pearson correlation:  +0.9231
  Y Pearson correlation:  +0.9343
  X mean abs error (nm):  2035
  Y mean abs error (nm):  2143

  ✓ Looks good! Target and actual deltas are well correlated.
Saved: .\hysteresis_dataset_reconstructed_updated.jsonl

## Segment split used for training/inspection

Confidence split (keeps contiguous high-confidence runs, threshold from `positions.json`):

`python ..\..\postprocess-stitch-recalculate\split_by_position_confidence.py .\hysteresis_dataset_reconstructed.jsonl .\positions.json --min-confidence 0.8 --output-dir .\confidence_0.8_segments`

Axis-correlation outliers were then removed and the affected segments split again to keep every JSONL contiguous:

`confidence_0.8_segments -> confidence_0.8_no_axis_outliers_segments`

Removed steps: `1736`, `2215`.
