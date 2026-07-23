| Level | Water condition | Registered | Mean features/img | 3D points | Mean reproj err (px) |
|-------|-----------------|------------|-------------------|-----------|----------------------|
| **baseline** | clear air (reference) | 30 / 30 (100%) | 12,698 | 12,269 | 0.527 |
| **I** | clearest open ocean | 30 / 30 (100%) | 12,744 | 12,111 | 0.545 |
| **II** | clear coastal water | 30 / 30 (100%) | 12,740 | 12,022 | 0.546 |
| **III** | turbid coastal water | 30 / 30 (100%) | 12,700 | 11,960 | 0.553 |
| **1C** | harbor (moderately turbid) | 30 / 30 (100%) | 13,181 | 12,178 | 0.567 |
| **3C** | harbor (very turbid) | 30 / 30 (100%) | 12,381 | 8,065 | 0.609 |
| **5C** | harbor (extremely turbid) | 24 / 30 (80%) | 10,599 | 961 | 0.641 |

### Color-correction ablation

| Level | Method | Registered | 3D points | Δ points vs uncorrected |
|-------|--------|------------|-----------|-------------------------|
| **3C** | sea_thru | 30/30 | 7,000 | -1,065 |
| **3C** | shades_of_gray | 30/30 | 3,994 | -4,071 |
| **5C** | sea_thru | 29/30 | 1,813 | +852 |
| **5C** | shades_of_gray | 4/30 | 115 | -846 |
