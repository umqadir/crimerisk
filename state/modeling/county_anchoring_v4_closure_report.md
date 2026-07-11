# County-Anchoring v4 Closure Evidence

Compared `state/candidates/county-anchoring-v4` against `state/candidates/step10-confidence-layer`.

## State Eta-Squared

| Geography | Field | Baseline | v4 | Delta |
|---|---:|---:|---:|---:|
| block_group | `index_total_primary_event_weighted` | 0.026674 | 0.032310 | +0.005636 |
| block_group | `index_total_part1_resident` | 0.014969 | 0.013652 | -0.001317 |
| tract | `index_total_primary_event_weighted` | 0.038199 | 0.040284 | +0.002085 |
| tract | `index_total_part1_resident` | 0.016254 | 0.016631 | +0.000377 |

## Rural County Dispersion

County-mean dispersion among rural block groups for KY/KS/NM/TX. Values below are `county_mean_std`; see CSV for p95-p05 and CV.

| State | Field | Baseline | v4 | Delta |
|---|---:|---:|---:|---:|
| KY | `index_total_primary_event_weighted` | 8.983508 | 12.287948 | +3.304441 |
| KY | `index_total_part1_resident` | 10.576317 | 14.048725 | +3.472408 |
| KS | `index_total_primary_event_weighted` | 18.250656 | 34.246269 | +15.995613 |
| KS | `index_total_part1_resident` | 17.773615 | 33.760283 | +15.986668 |
| NM | `index_total_primary_event_weighted` | 26.112951 | 40.911035 | +14.798084 |
| NM | `index_total_part1_resident` | 23.197947 | 37.315512 | +14.117565 |
| TX | `index_total_primary_event_weighted` | 166.028413 | 67.922624 | -98.105789 |
| TX | `index_total_part1_resident` | 484.562750 | 63.647779 | -420.914971 |

## Nonmunicipal Share

Nonmunicipal lane includes legacy `state_nonmunicipal_remainder` plus v4 `localized_remainder_*` county/residual components.
- baseline_step10: national all-offense nonmunicipal component share `0.156643` over count mass `1116491.924`.
- v4: national all-offense nonmunicipal component share `0.157514` over count mass `1115693.172`.
- Delta: `+0.000871`.

## Exposure-Floor Suppression

- block_group `primary_insufficient_exposure_rows`: baseline `0`, v4 `225`, delta `+225`.
- block_group `resident_insufficient_exposure_rows`: baseline `0`, v4 `600`, delta `+600`.
- tract `primary_insufficient_exposure_rows`: baseline `0`, v4 `60`, delta `+60`.
- tract `resident_insufficient_exposure_rows`: baseline `0`, v4 `165`, delta `+165`.

## Render Artifacts

PNG comparison renders are under `state/modeling/renders_v4/`.
