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

(.venv) PS C:\Users\wortelus\PycharmProjects\flexsem-data> .\.venv\Scripts\python.exe `
>>     .\postprocess-stitch-recalculate\split_by_position_confidence.py `                                      
>>     .\data_original\run71-data-feast-overnight\hysteresis_dataset_20260302_234218_clean_updated.jsonl `     
>>     .\data_original\run71-data-feast-overnight\positions.json `                                             
>>     --min-confidence 0.7 `                                                                                  
>>     --output-dir .\data_original\run71-data-feast-overnight\confidence_0.7_segments                         
Input format: json-array
Input records: 3036
Position entries: 3036
Minimum confidence: 0.7
Discarded records: 63
Contiguous high-confidence segments: 64
Segments to write: 64
Records to write: 2973
Output directory: data_original\run71-data-feast-overnight\confidence_0.7_segments
  hysteresis_dataset_20260302_234218_clean_updated_part001_steps0000-0055.jsonl: 56 records (steps 0..55)
  hysteresis_dataset_20260302_234218_clean_updated_part002_steps0057-0093.jsonl: 37 records (steps 57..93)
  hysteresis_dataset_20260302_234218_clean_updated_part003_steps0095-0177.jsonl: 83 records (steps 95..177)
  hysteresis_dataset_20260302_234218_clean_updated_part004_steps0179-0183.jsonl: 5 records (steps 179..183)
  hysteresis_dataset_20260302_234218_clean_updated_part005_steps0185-0202.jsonl: 18 records (steps 185..202)
  hysteresis_dataset_20260302_234218_clean_updated_part006_steps0204-0327.jsonl: 124 records (steps 204..327)
  hysteresis_dataset_20260302_234218_clean_updated_part007_steps0329-0342.jsonl: 14 records (steps 329..342)
  hysteresis_dataset_20260302_234218_clean_updated_part008_steps0344-0354.jsonl: 11 records (steps 344..354)
  hysteresis_dataset_20260302_234218_clean_updated_part009_steps0356-0372.jsonl: 17 records (steps 356..372)
  hysteresis_dataset_20260302_234218_clean_updated_part010_steps0374-0375.jsonl: 2 records (steps 374..375)
  hysteresis_dataset_20260302_234218_clean_updated_part011_steps0377-0432.jsonl: 56 records (steps 377..432)
  hysteresis_dataset_20260302_234218_clean_updated_part012_steps0434-0457.jsonl: 24 records (steps 434..457)
  hysteresis_dataset_20260302_234218_clean_updated_part013_steps0459-0620.jsonl: 162 records (steps 459..620)
  hysteresis_dataset_20260302_234218_clean_updated_part014_steps0622-0682.jsonl: 61 records (steps 622..682)
  hysteresis_dataset_20260302_234218_clean_updated_part015_steps0684-0755.jsonl: 72 records (steps 684..755)
  hysteresis_dataset_20260302_234218_clean_updated_part016_steps0757-0797.jsonl: 41 records (steps 757..797)
  hysteresis_dataset_20260302_234218_clean_updated_part017_steps0799-0823.jsonl: 25 records (steps 799..823)
  hysteresis_dataset_20260302_234218_clean_updated_part018_steps0825-0906.jsonl: 82 records (steps 825..906)
  hysteresis_dataset_20260302_234218_clean_updated_part019_steps0908-0921.jsonl: 14 records (steps 908..921)
  hysteresis_dataset_20260302_234218_clean_updated_part020_steps0923-1057.jsonl: 135 records (steps 923..1057)
  hysteresis_dataset_20260302_234218_clean_updated_part021_steps1059-1076.jsonl: 18 records (steps 1059..1076)
  hysteresis_dataset_20260302_234218_clean_updated_part022_steps1078-1112.jsonl: 35 records (steps 1078..1112)
  hysteresis_dataset_20260302_234218_clean_updated_part023_steps1114-1230.jsonl: 117 records (steps 1114..1230)
  hysteresis_dataset_20260302_234218_clean_updated_part024_steps1232-1234.jsonl: 3 records (steps 1232..1234)
  hysteresis_dataset_20260302_234218_clean_updated_part025_steps1236-1278.jsonl: 43 records (steps 1236..1278)
  hysteresis_dataset_20260302_234218_clean_updated_part026_steps1280-1314.jsonl: 35 records (steps 1280..1314)
  hysteresis_dataset_20260302_234218_clean_updated_part027_steps1316-1318.jsonl: 3 records (steps 1316..1318)
  hysteresis_dataset_20260302_234218_clean_updated_part028_steps1320-1334.jsonl: 15 records (steps 1320..1334)
  hysteresis_dataset_20260302_234218_clean_updated_part029_steps1336-1414.jsonl: 79 records (steps 1336..1414)
  hysteresis_dataset_20260302_234218_clean_updated_part030_steps1416-1431.jsonl: 16 records (steps 1416..1431)
  hysteresis_dataset_20260302_234218_clean_updated_part031_steps1433-1441.jsonl: 9 records (steps 1433..1441)
  hysteresis_dataset_20260302_234218_clean_updated_part032_steps1443-1452.jsonl: 10 records (steps 1443..1452)
  hysteresis_dataset_20260302_234218_clean_updated_part033_steps1454-1454.jsonl: 1 records (steps 1454..1454)
  hysteresis_dataset_20260302_234218_clean_updated_part034_steps1456-1461.jsonl: 6 records (steps 1456..1461)
  hysteresis_dataset_20260302_234218_clean_updated_part035_steps1463-1504.jsonl: 42 records (steps 1463..1504)
  hysteresis_dataset_20260302_234218_clean_updated_part036_steps1506-1541.jsonl: 36 records (steps 1506..1541)
  hysteresis_dataset_20260302_234218_clean_updated_part037_steps1543-1687.jsonl: 145 records (steps 1543..1687)
  hysteresis_dataset_20260302_234218_clean_updated_part038_steps1689-1714.jsonl: 26 records (steps 1689..1714)
  hysteresis_dataset_20260302_234218_clean_updated_part039_steps1716-1775.jsonl: 60 records (steps 1716..1775)
  hysteresis_dataset_20260302_234218_clean_updated_part040_steps1777-1780.jsonl: 4 records (steps 1777..1780)
  hysteresis_dataset_20260302_234218_clean_updated_part041_steps1782-1870.jsonl: 89 records (steps 1782..1870)
  hysteresis_dataset_20260302_234218_clean_updated_part042_steps1872-1904.jsonl: 33 records (steps 1872..1904)
  hysteresis_dataset_20260302_234218_clean_updated_part043_steps1906-1983.jsonl: 78 records (steps 1906..1983)
  hysteresis_dataset_20260302_234218_clean_updated_part044_steps1985-2108.jsonl: 124 records (steps 1985..2108)
  hysteresis_dataset_20260302_234218_clean_updated_part045_steps2110-2152.jsonl: 43 records (steps 2110..2152)
  hysteresis_dataset_20260302_234218_clean_updated_part046_steps2154-2162.jsonl: 9 records (steps 2154..2162)
  hysteresis_dataset_20260302_234218_clean_updated_part047_steps2164-2353.jsonl: 190 records (steps 2164..2353)
  hysteresis_dataset_20260302_234218_clean_updated_part048_steps2355-2392.jsonl: 38 records (steps 2355..2392)
  hysteresis_dataset_20260302_234218_clean_updated_part049_steps2394-2404.jsonl: 11 records (steps 2394..2404)
  hysteresis_dataset_20260302_234218_clean_updated_part050_steps2406-2491.jsonl: 86 records (steps 2406..2491)
  hysteresis_dataset_20260302_234218_clean_updated_part051_steps2493-2531.jsonl: 39 records (steps 2493..2531)
  hysteresis_dataset_20260302_234218_clean_updated_part052_steps2533-2566.jsonl: 34 records (steps 2533..2566)
  hysteresis_dataset_20260302_234218_clean_updated_part053_steps2568-2607.jsonl: 40 records (steps 2568..2607)
  hysteresis_dataset_20260302_234218_clean_updated_part054_steps2609-2621.jsonl: 13 records (steps 2609..2621)
  hysteresis_dataset_20260302_234218_clean_updated_part055_steps2623-2631.jsonl: 9 records (steps 2623..2631)
  hysteresis_dataset_20260302_234218_clean_updated_part056_steps2633-2671.jsonl: 39 records (steps 2633..2671)
  hysteresis_dataset_20260302_234218_clean_updated_part057_steps2673-2688.jsonl: 16 records (steps 2673..2688)
  hysteresis_dataset_20260302_234218_clean_updated_part058_steps2690-2762.jsonl: 73 records (steps 2690..2762)
  hysteresis_dataset_20260302_234218_clean_updated_part059_steps2764-2820.jsonl: 57 records (steps 2764..2820)
  hysteresis_dataset_20260302_234218_clean_updated_part060_steps2822-2904.jsonl: 83 records (steps 2822..2904)
  hysteresis_dataset_20260302_234218_clean_updated_part061_steps2906-2974.jsonl: 69 records (steps 2906..2974)
  hysteresis_dataset_20260302_234218_clean_updated_part062_steps2976-2996.jsonl: 21 records (steps 2976..2996)
  hysteresis_dataset_20260302_234218_clean_updated_part063_steps2998-3031.jsonl: 34 records (steps 2998..3031)
  hysteresis_dataset_20260302_234218_clean_updated_part064_steps3033-3035.jsonl: 3 records (steps 3033..3035)
Wrote 64 segments and data_original\run71-data-feast-overnight\confidence_0.7_segments\manifest.json
