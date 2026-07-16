Because this run crashed before the full `.jsonl` hysteresis dataset
could be saved, I made a `reconstruct_hysteresis_jsonl_from_logs.py` script to reconstruct it.

Also, from the generated sub0-sub2 sessions I removed some outliers, which are noted
in the `run70_filtered_axis_outliers_iqr.csv` file.