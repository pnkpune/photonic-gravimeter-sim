# Norwegian Margin Regional Benchmark Report

## Scope

This report compares the existing synthetic maritime benchmark against the new Norwegian-margin regional gravity benchmark path.

- regional benchmark summary: `data/outputs/reports/norwegian_margin_benchmark/norwegian_margin_maritime_summary.json`
- synthetic benchmark summary: `data/outputs/reports/priority3_benchmark/maritime_baseline_summary.json`
- regional manifest: `data/gravity_maps/processed/norwegian_margin_gravity_map_manifest.json`
- comparison figure: `data/outputs/figures/norwegian_margin_regional_benchmark/norwegian_margin_rmse_comparison.png`

## Regional Map Source

- region: `norwegian_margin`
- source name: `bundled_fixture`
- source kind: `regular_grid_csv`
- raw data path: `data/gravity_maps/raw/norwegian_margin/norwegian_margin_fixture.csv`
- processed map path: `data/gravity_maps/processed/norwegian_margin_gravity_map.npz`
- latitude bounds [deg]: `63.580` to `63.820`
- longitude bounds [deg]: `4.000` to `4.480`
- grid shape: `7 x 9`
- spacing [deg]: `0.040` lat, `0.060` lon

Important note: the current in-repo Norwegian-margin input is a small tracked fixture that exercises the real regional ingest/cache path. It is not a full public survey grid.

## Regional Benchmark

| Configuration | INS RMSE [m] | INS CEP95 [m] | PF RMSE [m] | Sequence RMSE [m] | Lag-smoothed RMSE [m] | HMI horiz [%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ins_only` | 111.474 | 196.457 | n/a | n/a | n/a | 0.000 |
| `observe_only` | 111.474 | 196.457 | 133.955 | n/a | n/a | 0.000 |
| `observe_plus_gradient` | 111.474 | 196.457 | 131.762 | n/a | n/a | 0.000 |
| `sequence_plus_gradient` | 111.474 | 196.457 | n/a | 110.219 | n/a | 0.000 |
| `sequence_plus_gradient_lag_smoothed` | 111.474 | 196.457 | n/a | 110.219 | 110.763 | 0.000 |

## Synthetic vs Regional Comparison

| Estimator path | Synthetic RMSE [m] | Regional RMSE [m] |
| --- | ---: | ---: |
| live INS | 103.512 | 111.474 |
| best PF observe-only | 96.902 | 131.762 |
| best sequence observe-only | 92.808 | 110.219 |
| best lag-smoothed output | 98.268 | 110.763 |

## Findings

- Sequence still beats PF on the Norwegian regional benchmark: `True`.
- The bounded-lag smoother still beats the live INS on the Norwegian regional benchmark with zero lag-output HMI: `True`.
- Synthetic best sequence improvement vs live INS: `10.34%`.
- Regional best sequence improvement vs live INS: `1.13%`.
- Synthetic best lag-smoothed improvement vs live INS: `5.07%`.
- Regional best lag-smoothed improvement vs live INS: `0.64%`.
- Gains shrink materially relative to the synthetic benchmark: `True`.

## Conclusion

The current estimator ranking survives on the Norwegian-margin fixture path: sequence-based matching remains better than PF, and the bounded-lag smoother still beats the live INS. The improvement is much smaller than on the synthetic maritime benchmark, so the main lesson is not that the algorithm failed, but that the synthetic map materially overstated how much gravity distinctiveness was available.
