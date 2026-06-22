Because I mistyped the max timeout retry in the experiment (swapped secs / millisecs),
I had to cut off the experiment, so use the clean version pls :)

positions.json are calculated from
python .\composite_positions.py .\run71-data-feast-overnight-2\stitchedv3_full.png .\run71-data-feast-overnight-2\temp --crop-bottom 60
against the `stitchedv3` from stitch.py

positions.json are also from the clean version (3036 positions)
there are 80 positions with confidence < 0.8 for `"confidence":\s*0\.[0-7]\d*`
i think we should just disregard them (and probably separate create experiment file for each one of the normal confidence sections)

full_verified goes from verify_positions.py

hysteresis_dataset.....234218.jsonl is the original one with all (even later unwanted) steps