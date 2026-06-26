`python ..\..\postprocess-stitch-recalculate\update_positions.py .\hysteresis_dataset_reconstructed.jsonl .\positions.json --ref-step 4379`     

Updated: 1900, Skipped: 0

============================================================
SANITY CHECK: target delta vs. actual delta direction
============================================================
  X direction agreement: 1885/1899 (99%)
  Y direction agreement: 1578/1899 (83%)
  X Pearson correlation:  +0.9468
  Y Pearson correlation:  +0.9265
  X mean abs error (nm):  2585
  Y mean abs error (nm):  3524

  ✓ Looks good! Target and actual deltas are well correlated.
Saved: .\hysteresis_dataset_reconstructed_updated.jsonl