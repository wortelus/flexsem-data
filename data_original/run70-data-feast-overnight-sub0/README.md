`data_original\run70-data-feast-overnight-sub0> python ..\..\postprocess-stitch-recalculate\update_positions.py .\hysteresis_dataset_reconstructed.jsonl .\positions.json --ref-step 1`     

Updated: 1637, Skipped: 0

============================================================
SANITY CHECK: target delta vs. actual delta direction
============================================================
  X direction agreement: 1393/1636 (85%)
  Y direction agreement: 1450/1636 (89%)
  X Pearson correlation:  +0.9466
  Y Pearson correlation:  +0.9500
  X mean abs error (nm):  2167
  Y mean abs error (nm):  1690

  ✓ Looks good! Target and actual deltas are well correlated.
Saved: .\hysteresis_dataset_reconstructed_updated.jsonl

## Segment split used for training/inspection

Confidence split (keeps contiguous high-confidence runs, threshold from `positions.json`):

`python ..\..\postprocess-stitch-recalculate\split_by_position_confidence.py .\hysteresis_dataset_reconstructed.jsonl .\positions.json --min-confidence 0.8 --output-dir .\confidence_0.8_segments`

Axis-correlation outliers were then removed and the affected segments split again to keep every JSONL contiguous:

`confidence_0.8_segments -> confidence_0.8_no_axis_outliers_segments`

Removed steps: `357`, `1105`.
