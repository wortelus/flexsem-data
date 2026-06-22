data commited here are from the old repo, however
the data were checked against verify_positions.py
and the positions checks out

i also checked the first ~500 samples in verify/ (verify_positions.py) and it seems all of them check out
i think there is only with confidence ~0.8

„clean“ -- data_original/run55-random-walk-20um/hysteresis_dataset_20260210_125219_updated.jsonl, exact match:

- 1003 records
- 1003 positions
- 0 missing img_path references
- 1003/1003 actual values
- max diff: 0 nm

eq:

x_actual_abs = round(-17250 - (x_px - 1897) × 12.40234)
y_actual_abs = round(-37800 + (y_px - 1776) × 12.40234)

Reference is set to 0