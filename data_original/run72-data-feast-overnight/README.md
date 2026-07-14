here, all the data are clean, so no need to clean them


```
python .\postprocess-stitch-recalculate\update_positions.py .\data_original\run72-data-feast-overnight\hysteresis_dataset_20260303_203815.jsonl .\data_original\run72-data-feast-overnight\positions.json --output .\data_original\run72-data-feast-overnight\hysteresis_dataset_20260303_203815_updated.jsonl

Updated: 9056, Skipped: 0

============================================================
SANITY CHECK: target delta vs. actual delta direction
============================================================
  X direction agreement: 7342/9050 (81%)
  Y direction agreement: 7219/9050 (80%)
  X Pearson correlation:  +0.9541
  Y Pearson correlation:  +0.9566
  X mean abs error (nm):  1927
  Y mean abs error (nm):  2391

  ✓ Looks good! Target and actual deltas are well correlated.
Saved: .\data_original\run72-data-feast-overnight\hysteresis_dataset_20260303_203815_updated.jsonl
(.venv) PS C:\Users\wortelus\PycharmProjects\flexsem-data>
```