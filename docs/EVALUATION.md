# Evaluation

## Run

| Parameter | Value |
|---|---|
| Run ID | `gold_v53r2` |
| Build year | 2025 |
| Candidate | `state/candidates/v53-2025` |
| Spatial folds | 5 leave-one-feed-city-out: the AGS feed-free cities San Francisco, Washington, Denver, Minneapolis and St. Louis. Four cities score each offense except rape, which has one |
| Temporal folds | Feed-truncation experiments, not historical forecasts: feeds truncated at 2021 and at 2023, everything else 2025 |
| Model fold | not rerun for 2025.1; the `gold_v1` model rows below remain the reference |
| Bootstrap | 1,000 city-cluster resamples, seed 20260921 |
| Fold wall time | 2,981.3 seconds |
| Attempt wall time | 2,986.3 seconds |
| Results | `state/eval/gold_v53r2/results.csv` |
| Calibration | `state/eval/gold_v53r2/calibration.csv` |
| Per-cell inputs | `state/eval/gold_v53r2/inputs/` |
| Run manifest | `state/eval/gold_v53r2/run_manifest.json` |

`gold_v53r2` scores the code and artifacts the 2025.1 surface is built from, with the
stage-10 normalizer surface and the stage-11 expert table retrained from current
inputs rather than pinned (stage numbers as in `docs/BUILD.md`). The spatial folds are restricted to the five cities AGS
does not take a feed from, so `ours` and AGS are compared on the same held-out ground.
Every fold surface was deleted after scoring. At the end of the run the harness
re-hashed the shared city-feed artifacts and the other `state/modeling` parquets it
read: 18 protected paths and 73 observed paths, none changed, added or removed
(`protected_state` in the run manifest).

Against `gold_v53`, the pinned-artifact run below, the retrain moves held-out TVD by
at most 0.010 on any offense and fold. The largest move is spatial motor vehicle theft,
0.288 to 0.298; murder rises by 0.003 or less on all three folds and aggravated assault
falls by 0.002 or less. Every baseline arm is unchanged, because only the allocator's
expert table was retrained.

## Baseline arms

TVD values are city means; lower is better. `ours` uses published support: tracts for
murder and rape, block groups otherwise. `ours`, population and primary exposure in a
row are scored on the same cities and cells. Past counts use the training window only
and are scored only in the cities that have them (Past-counts n); the last column is
`ours` on those cities. AGS is reported in the tract table below.

| Fold | Offense | n cities | Ours TVD | Population TVD | Primary-exposure TVD | Past-counts n | Past-counts TVD | Ours TVD, past-counts cities |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| spatial | aggravated_assault | 4 | 0.357 | 0.449 | 0.423 | — | — | — |
| spatial | burglary | 4 | 0.287 | 0.321 | 0.280 | — | — | — |
| spatial | larceny | 4 | 0.255 | 0.384 | 0.270 | — | — | — |
| spatial | motor_vehicle_theft | 4 | 0.298 | 0.287 | 0.332 | — | — | — |
| spatial | murder | 4 | 0.471 | 0.602 | 0.601 | — | — | — |
| spatial | rape | 1 | 0.461 | 0.479 | 0.469 | — | — | — |
| spatial | robbery | 4 | 0.409 | 0.475 | 0.438 | — | — | — |
| temporal_2021 | aggravated_assault | 8 | 0.347 | 0.494 | 0.474 | 4 | 0.356 | 0.366 |
| temporal_2021 | burglary | 10 | 0.293 | 0.399 | 0.352 | 7 | 0.296 | 0.291 |
| temporal_2021 | larceny | 11 | 0.228 | 0.399 | 0.300 | 7 | 0.251 | 0.221 |
| temporal_2021 | motor_vehicle_theft | 10 | 0.310 | 0.388 | 0.430 | 7 | 0.311 | 0.316 |
| temporal_2021 | murder | 7 | 0.476 | 0.618 | 0.615 | 4 | 0.441 | 0.436 |
| temporal_2021 | rape | 1 | 0.461 | 0.479 | 0.469 | — | — | — |
| temporal_2021 | robbery | 10 | 0.370 | 0.526 | 0.491 | 6 | 0.355 | 0.350 |
| temporal_2023 | aggravated_assault | 8 | 0.364 | 0.501 | 0.482 | 4 | 0.377 | 0.399 |
| temporal_2023 | burglary | 10 | 0.330 | 0.427 | 0.384 | 7 | 0.345 | 0.344 |
| temporal_2023 | larceny | 11 | 0.245 | 0.410 | 0.314 | 7 | 0.259 | 0.248 |
| temporal_2023 | motor_vehicle_theft | 10 | 0.337 | 0.416 | 0.453 | 7 | 0.347 | 0.353 |
| temporal_2023 | murder | 7 | 0.534 | 0.656 | 0.653 | 4 | 0.541 | 0.538 |
| temporal_2023 | rape | 1 | 0.461 | 0.479 | 0.469 | — | — | — |
| temporal_2023 | robbery | 10 | 0.409 | 0.550 | 0.518 | 6 | 0.406 | 0.415 |

Rows where a baseline has lower TVD than `ours`:

| Fold | Offense | Baseline | Baseline TVD | Ours TVD |
|---|---|---|---:|---:|
| spatial | motor_vehicle_theft | Population | 0.287 | 0.298 |
| spatial | burglary | Primary exposure | 0.280 | 0.287 |
| temporal_2021 | aggravated_assault | Past counts, same cities | 0.356 | 0.366 |
| temporal_2021 | motor_vehicle_theft | Past counts, same cities | 0.311 | 0.316 |
| temporal_2023 | aggravated_assault | Past counts, same cities | 0.377 | 0.399 |
| temporal_2023 | motor_vehicle_theft | Past counts, same cities | 0.347 | 0.353 |
| temporal_2023 | robbery | Past counts, same cities | 0.406 | 0.415 |

Past counts are not scored in Denver, Minneapolis or St. Louis in either temporal fold,
nor in Baltimore for aggravated assault, larceny and robbery.

## Release evaluation check

`_check_release_evaluation` in `scripts/diagnostics/validate_release_outputs.py`.

| Requirement | Status in 2025.1.1 | Result |
|---|---|---|
| `gold_v53r2` results and manifest present, run id and build year match | Blocking | Pass |
| Evaluated candidate equals the published candidate (`v53-2025`) | Blocking | Pass |
| Spatial TVD: `ours` below population for every offense except rape | Reported, not blocking (`RELEASE_GOLD_TVD_BLOCKING = False`) | Fail: motor vehicle theft, 0.2983 against 0.2873 |

Rape (one eligible city) is reported and not gated: 0.461 against 0.479.

## AGS 2022A tract comparison

`ours_tract` is the sum of block-group expected counts on the exact tract universe used by AGS. AGS uses city incident feeds for New York, Chicago, Boston, Philadelphia, Baltimore, Seattle, Austin, and Mesa; none of those cities is a spatial fold here, so every row is `ags_uses_city_feed=false`. Values are estimate [city-resampled 95% interval].

| Offense | AGS city feed | n cities | Ours tract TVD | AGS TVD | Ours tract ρ | AGS ρ | Ours tract W1 skill | AGS W1 skill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| murder | false | 4 | 0.471 [0.331, 0.631] | 0.449 [0.267, 0.595] | 0.498 [0.363, 0.616] | 0.540 [0.299, 0.749] | 0.473 [0.316, 0.631] | 0.515 [0.209, 0.738] |
| rape | false | 1 | 0.461 [NA] | 0.481 [NA] | 0.086 [NA] | 0.180 [NA] | 0.086 [NA] | 0.153 [NA] |
| robbery | false | 4 | 0.293 [0.267, 0.320] | 0.295 [0.271, 0.323] | 0.609 [0.569, 0.636] | 0.595 [0.565, 0.620] | 0.463 [0.323, 0.602] | 0.462 [0.223, 0.621] |
| aggravated_assault | false | 4 | 0.263 [0.220, 0.314] | 0.266 [0.193, 0.347] | 0.680 [0.608, 0.733] | 0.656 [0.542, 0.730] | 0.501 [0.246, 0.710] | 0.558 [0.442, 0.673] |
| burglary | false | 4 | 0.212 [0.196, 0.242] | 0.227 [0.186, 0.263] | 0.505 [0.402, 0.581] | 0.391 [0.317, 0.491] | 0.254 [-0.279, 0.622] | 0.008 [-0.502, 0.359] |
| larceny | false | 4 | 0.195 [0.156, 0.234] | 0.250 [0.178, 0.321] | 0.741 [0.682, 0.805] | 0.634 [0.548, 0.731] | 0.544 [0.369, 0.713] | 0.525 [0.369, 0.681] |
| motor_vehicle_theft | false | 4 | 0.219 [0.206, 0.232] | 0.214 [0.186, 0.234] | 0.634 [0.550, 0.706] | 0.620 [0.582, 0.646] | 0.247 [0.098, 0.396] | 0.173 [0.093, 0.292] |

## Interval calibration

Held-out coverage of the internal intervals is below nominal at every level in this
table (50, 80 and 95), so 2025.1 publishes no interval and no reliability tier; the
uncertainty layer is retained internally and the factors below are not applied to any
surface.

Each cell reports empirical coverage [95% interval] / log-width factor [95% interval]. Factors multiply each existing asymmetric log-share half-width about the point estimate. The transform is `log(share + 0.5 / city incidents)`. Factors are fitted on held-out spatial cities with equal city weight. Direct and benchmark classes have no held-out spatial cells after feed-city exclusion and are reported as NA.

| Offense | Support class | n cities | n cells (50/80/95) | 50 coverage / factor | 80 coverage / factor | 95 coverage / factor |
|---|---|---:|---:|---:|---:|---:|
| murder | model | 4 | 609/609/609 | 0.105 [0.029, 0.200] / 4.018 [3.317, 4.538] | 0.176 [0.035, 0.317] / 3.528 [2.726, 4.275] | 0.476 [0.270, 0.677] / 1.079 [1.003, 1.159] |
| murder | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| murder | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| rape | model | 1 | 104/104/104 | 0.087 [0.087, 0.087] / 5.010 [5.010, 5.010] | 0.240 [0.240, 0.240] / 4.002 [4.002, 4.002] | 0.587 [0.587, 0.587] / 1.321 [1.321, 1.321] |
| rape | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| rape | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| robbery | model | 4 | 1943/1961/1943 | 0.145 [0.106, 0.205] / 2.759 [2.276, 3.564] | 0.425 [0.336, 0.537] / 1.875 [1.776, 1.989] | 0.616 [0.525, 0.712] / 1.832 [1.644, 1.993] |
| robbery | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| robbery | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| aggravated_assault | model | 4 | 1945/1961/1945 | 0.184 [0.159, 0.209] / 2.690 [2.325, 3.020] | 0.530 [0.492, 0.585] / 1.666 [1.503, 1.820] | 0.606 [0.535, 0.689] / 2.670 [2.261, 2.847] |
| aggravated_assault | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| aggravated_assault | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| burglary | model | 4 | 1941/1961/1941 | 0.155 [0.143, 0.169] / 5.303 [3.895, 8.331] | 0.563 [0.507, 0.609] / 1.677 [1.519, 1.938] | 0.741 [0.707, 0.782] / 2.052 [1.732, 2.314] |
| burglary | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| burglary | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| larceny | model | 4 | 1949/1961/1949 | 0.310 [0.263, 0.354] / 1.471 [1.089, 3.260] | 0.777 [0.736, 0.826] / 1.072 [0.956, 1.240] | 0.528 [0.444, 0.614] / 4.277 [3.084, 5.123] |
| larceny | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| larceny | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| motor_vehicle_theft | model | 4 | 1950/1961/1950 | 0.153 [0.111, 0.199] / inf [4.818, inf] | 0.587 [0.530, 0.640] / 1.720 [1.526, 1.922] | 0.677 [0.618, 0.767] / 2.565 [2.203, 2.692] |
| motor_vehicle_theft | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| motor_vehicle_theft | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |

## Protocol

Spatial folds remove one admitted feed city from direct allocation, residual training, and uncertainty training. Held-out pooled 2018–2024 incidents supply truth.

Temporal folds truncate the city incident feeds at 2021 or 2023 and score later
incidents. Every other fold input (jurisdiction totals, denominators, covariates,
admission and repair rules, tuned constants) is the 2025 edition's. They are
feed-truncation experiments, not historical forecasts; no input has been rebuilt as of
a forecast origin.

`gold_v53r2` runs no model fold; mixture weights v3, soft shrinkage, the rape triple, exposure ensemble weights, murder K and tau are the constants selected on the `gold_v1` model corpus and are named in `reuse_flags` when reused.

Metrics are TVD, Spearman correlation, top-decile capture, exact spatial Wasserstein skill against the population null, and interval coverage. Confidence intervals resample cities. Negative skill and losses remain in the output.

## What the spatial evidence covers

Four cities per offense, one for rape. Not measured:

- Transfer outside the feed cities. Feed-city performance is not a bound on
  performance elsewhere.
- Allocation outside the 13 feed cities: about 95% of block groups.
- 98.1% of the edition's block groups outside the jurisdiction-panel training hull.
- Direct and benchmark interval calibration under spatial holdout.
- Spatial rape performance beyond one eligible feed city.
- Spatial performance in the eight cities AGS also takes a feed from; `gold_v1` below covers those.
- Jurisdiction totals, the national-relative indexes and the Overall categories. Gold
  scores within-city shares only.
- Suburban, small-city and rural allocation: no incident dataset outside the 13 feed
  cities is scored.

## Previous candidate: `gold_v53`, the pinned-artifact build

`gold_v53` is the same seven folds on the same code, scored before the stage-10
normalizer surface and the stage-11 expert table were retrained. Only the held-out TVD table
is kept here; its full results, AGS comparison and interval calibration remain in
`state/eval/gold_v53/`. Fold wall time was 3,032.4 seconds.

| Fold | Offense | n cities | Ours TVD | Population TVD | Primary-exposure TVD | Past-counts TVD |
|---|---|---:|---:|---:|---:|---:|
| spatial | aggravated_assault | 4 | 0.358 | 0.449 | 0.423 | — |
| spatial | burglary | 4 | 0.287 | 0.321 | 0.280 | — |
| spatial | larceny | 4 | 0.255 | 0.384 | 0.270 | — |
| spatial | motor_vehicle_theft | 4 | 0.288 | 0.287 | 0.332 | — |
| spatial | murder | 4 | 0.467 | 0.602 | 0.601 | — |
| spatial | rape | 1 | 0.460 | 0.479 | 0.469 | — |
| spatial | robbery | 4 | 0.409 | 0.475 | 0.438 | — |
| temporal_2021 | aggravated_assault | 8 | 0.348 | 0.494 | 0.474 | 0.356 |
| temporal_2021 | burglary | 10 | 0.293 | 0.399 | 0.352 | 0.296 |
| temporal_2021 | larceny | 11 | 0.228 | 0.399 | 0.300 | 0.251 |
| temporal_2021 | motor_vehicle_theft | 10 | 0.305 | 0.388 | 0.430 | 0.311 |
| temporal_2021 | murder | 7 | 0.474 | 0.618 | 0.615 | 0.441 |
| temporal_2021 | rape | 1 | 0.460 | 0.479 | 0.469 | — |
| temporal_2021 | robbery | 10 | 0.370 | 0.526 | 0.491 | 0.355 |
| temporal_2023 | aggravated_assault | 8 | 0.365 | 0.501 | 0.482 | 0.377 |
| temporal_2023 | burglary | 10 | 0.330 | 0.427 | 0.384 | 0.345 |
| temporal_2023 | larceny | 11 | 0.246 | 0.410 | 0.314 | 0.259 |
| temporal_2023 | motor_vehicle_theft | 10 | 0.333 | 0.416 | 0.453 | 0.347 |
| temporal_2023 | murder | 7 | 0.532 | 0.656 | 0.653 | 0.541 |
| temporal_2023 | rape | 1 | 0.460 | 0.479 | 0.469 | — |
| temporal_2023 | robbery | 10 | 0.409 | 0.550 | 0.518 | 0.406 |

Every difference against the release table above is at most 0.010 TVD, on spatial
motor vehicle theft, and the published national anchors, denominators and cell counts
are identical between the two builds. The retrain is a reproducibility fix; it is not
an accuracy improvement and is not offered as one.

## Reference: `gold_v1`, before the footprint and level-lane fixes

`gold_v1` is the 13-city run on the pre-fix surface. It is kept for comparison; the
tables above are the 2025.1 release numbers.

### Run

| Parameter | Value |
|---|---|
| Run ID | `gold_v1` |
| Build year | 2025 |
| Spatial folds | 13 leave-one-feed-city-out |
| Temporal folds | feeds through 2021; feeds through 2023 |
| Model fold | E4 base variant; 20 sources; 40 jurisdictions; 190 cells |
| Bootstrap | 1,000 city-cluster resamples |
| Fold wall time | 13,031.2 seconds |
| Attempt wall time | 13,050.3 seconds |
| Results | `state/eval/gold_v1/results.csv` |
| Calibration | `state/eval/gold_v1/calibration.csv` |
| Per-cell inputs | `state/eval/gold_v1/inputs/` |
| Run manifest | `state/eval/gold_v1/run_manifest.json` |

### Baseline arms

TVD values are city means. `ours` uses published support: tracts for murder and rape, block groups otherwise. Past counts use the training window only. AGS is reported in the tract table below.

| Fold | Offense | n cities | Ours TVD | Population TVD | Primary-exposure TVD | Past-counts TVD |
|---|---|---:|---:|---:|---:|---:|
| model | aggravated_assault | 18 | 0.287 | 0.375 | 0.361 | — |
| model | burglary | 20 | 0.262 | 0.330 | 0.300 | — |
| model | larceny | 20 | 0.318 | 0.424 | 0.330 | — |
| model | motor_vehicle_theft | 20 | 0.268 | 0.329 | 0.348 | — |
| model | murder | 16 | 0.342 | 0.433 | 0.424 | — |
| model | rape | 10 | 0.293 | 0.316 | 0.296 | — |
| model | robbery | 20 | 0.316 | 0.415 | 0.393 | — |
| spatial | aggravated_assault | 9 | 0.342 | 0.434 | 0.421 | — |
| spatial | burglary | 11 | 0.287 | 0.338 | 0.289 | — |
| spatial | larceny | 12 | 0.258 | 0.377 | 0.274 | — |
| spatial | motor_vehicle_theft | 11 | 0.306 | 0.318 | 0.370 | — |
| spatial | murder | 7 | 0.440 | 0.555 | 0.552 | — |
| spatial | rape | 1 | 0.460 | 0.479 | 0.469 | — |
| spatial | robbery | 11 | 0.395 | 0.462 | 0.433 | — |
| temporal_2021 | aggravated_assault | 9 | 0.281 | 0.434 | 0.420 | 0.156 |
| temporal_2021 | burglary | 11 | 0.243 | 0.351 | 0.302 | 0.192 |
| temporal_2021 | larceny | 12 | 0.208 | 0.386 | 0.283 | 0.144 |
| temporal_2021 | motor_vehicle_theft | 11 | 0.249 | 0.329 | 0.380 | 0.205 |
| temporal_2021 | murder | 7 | 0.414 | 0.570 | 0.567 | 0.344 |
| temporal_2021 | rape | 1 | 0.460 | 0.479 | 0.469 | — |
| temporal_2021 | robbery | 11 | 0.311 | 0.467 | 0.438 | 0.223 |
| temporal_2023 | aggravated_assault | 9 | 0.299 | 0.443 | 0.429 | 0.201 |
| temporal_2023 | burglary | 11 | 0.284 | 0.383 | 0.339 | 0.255 |
| temporal_2023 | larceny | 12 | 0.221 | 0.395 | 0.296 | 0.167 |
| temporal_2023 | motor_vehicle_theft | 11 | 0.271 | 0.359 | 0.405 | 0.249 |
| temporal_2023 | murder | 7 | 0.503 | 0.639 | 0.635 | 0.480 |
| temporal_2023 | rape | 1 | 0.460 | 0.479 | 0.469 | — |
| temporal_2023 | robbery | 11 | 0.349 | 0.492 | 0.467 | 0.288 |

### AGS 2022A tract comparison

`ours_tract` is the sum of block-group expected counts on the exact tract universe used by AGS. AGS uses city incident feeds for New York, Chicago, Boston, Philadelphia, Baltimore, Seattle, Austin, and Mesa. The feed-free cities are San Francisco, Washington, Denver, Minneapolis, and St. Louis. Values are estimate [city-resampled 95% interval].

| Offense | AGS city feed | n cities | Ours tract TVD | AGS TVD | Ours tract ρ | AGS ρ | Ours tract W1 skill | AGS W1 skill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| murder | false | 4 | 0.467 [0.337, 0.617] | 0.449 [0.267, 0.595] | 0.512 [0.383, 0.631] | 0.540 [0.299, 0.749] | 0.453 [0.294, 0.611] | 0.515 [0.209, 0.738] |
| murder | true | 3 | 0.403 [0.350, 0.480] | 0.348 [0.181, 0.444] | 0.483 [0.367, 0.619] | 0.637 [0.481, 0.887] | 0.596 [0.555, 0.639] | 0.514 [0.056, 0.871] |
| rape | false | 1 | 0.460 [NA] | 0.481 [NA] | 0.081 [NA] | 0.180 [NA] | 0.091 [NA] | 0.153 [NA] |
| robbery | false | 5 | 0.299 [0.278, 0.321] | 0.296 [0.273, 0.318] | 0.570 [0.507, 0.621] | 0.587 [0.563, 0.614] | 0.430 [0.298, 0.572] | 0.414 [0.239, 0.588] |
| robbery | true | 6 | 0.292 [0.271, 0.313] | 0.255 [0.193, 0.328] | 0.642 [0.602, 0.676] | 0.733 [0.622, 0.819] | 0.550 [0.492, 0.625] | 0.503 [0.268, 0.685] |
| aggravated_assault | false | 5 | 0.271 [0.234, 0.313] | 0.273 [0.219, 0.338] | 0.679 [0.621, 0.727] | 0.666 [0.570, 0.725] | 0.522 [0.317, 0.656] | 0.591 [0.472, 0.700] |
| aggravated_assault | true | 4 | 0.231 [0.215, 0.247] | 0.162 [0.124, 0.210] | 0.724 [0.696, 0.758] | 0.846 [0.770, 0.909] | 0.549 [0.461, 0.629] | 0.717 [0.538, 0.857] |
| burglary | false | 5 | 0.230 [0.196, 0.271] | 0.248 [0.201, 0.298] | 0.480 [0.401, 0.557] | 0.413 [0.338, 0.489] | 0.245 [-0.182, 0.556] | 0.018 [-0.390, 0.293] |
| burglary | true | 6 | 0.202 [0.181, 0.219] | 0.211 [0.160, 0.262] | 0.666 [0.617, 0.722] | 0.683 [0.586, 0.772] | 0.481 [0.365, 0.573] | 0.387 [0.157, 0.593] |
| larceny | false | 5 | 0.202 [0.166, 0.232] | 0.258 [0.194, 0.316] | 0.728 [0.678, 0.780] | 0.616 [0.539, 0.702] | 0.495 [0.326, 0.664] | 0.477 [0.321, 0.634] |
| larceny | true | 7 | 0.198 [0.172, 0.224] | 0.200 [0.160, 0.246] | 0.705 [0.653, 0.760] | 0.742 [0.656, 0.816] | 0.514 [0.419, 0.605] | 0.578 [0.307, 0.757] |
| motor_vehicle_theft | false | 5 | 0.233 [0.204, 0.272] | 0.224 [0.197, 0.248] | 0.606 [0.517, 0.695] | 0.628 [0.593, 0.653] | 0.219 [0.068, 0.354] | 0.229 [0.110, 0.355] |
| motor_vehicle_theft | true | 6 | 0.233 [0.215, 0.252] | 0.229 [0.172, 0.291] | 0.640 [0.596, 0.691] | 0.692 [0.607, 0.766] | 0.410 [0.208, 0.631] | 0.403 [0.124, 0.645] |

### Interval calibration

Held-out coverage of the internal intervals is below nominal at every level in the table below (50, 80 and 95), so 2025.1 publishes no interval and no reliability tier; the layer is retained internally and the factors here are not applied to any surface.

Each cell reports empirical coverage [95% interval] / log-width factor [95% interval]. Factors multiply each existing asymmetric log-share half-width about the point estimate. The transform is `log(share + 0.5 / city incidents)`. Factors are fitted on held-out spatial cities with equal city weight and are not applied to any surface. Direct and benchmark classes have no held-out spatial cells after feed-city exclusion and are reported as NA.

| Offense | Support class | n cities | n cells (50/80/95) | 50 coverage / factor | 80 coverage / factor | 95 coverage / factor |
|---|---|---:|---:|---:|---:|---:|
| murder | model | 7 | 3828/3906/3828 | 0.114 [0.068, 0.171] / 3.789 [3.384, 4.028] | 0.221 [0.108, 0.331] / 3.503 [2.854, 4.030] | 0.591 [0.401, 0.777] / 1.012 [1.000, 1.062] |
| murder | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| murder | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| rape | model | 1 | 104/104/104 | 0.106 [0.106, 0.106] / 4.912 [4.912, 4.912] | 0.250 [0.250, 0.250] / 3.694 [3.694, 3.694] | 0.587 [0.587, 0.587] / 1.300 [1.300, 1.300] |
| rape | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| rape | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| robbery | model | 11 | 13165/13579/13165 | 0.170 [0.141, 0.200] / 2.819 [2.431, 3.294] | 0.478 [0.416, 0.541] / 1.808 [1.700, 1.893] | 0.700 [0.630, 0.767] / 1.761 [1.513, 1.971] |
| robbery | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| robbery | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| aggravated_assault | model | 9 | 13015/13457/13015 | 0.198 [0.167, 0.225] / 2.595 [2.397, 2.932] | 0.556 [0.479, 0.617] / 1.642 [1.475, 1.801] | 0.665 [0.569, 0.752] / 2.271 [1.953, 2.502] |
| aggravated_assault | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| aggravated_assault | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| burglary | model | 11 | 13766/14299/13766 | 0.164 [0.142, 0.185] / 4.523 [3.824, 5.498] | 0.576 [0.510, 0.633] / 1.723 [1.510, 1.943] | 0.806 [0.744, 0.864] / 1.600 [1.426, 1.857] |
| burglary | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| burglary | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| larceny | model | 12 | 13753/14160/13753 | 0.316 [0.296, 0.336] / 1.492 [1.265, 1.948] | 0.768 [0.747, 0.787] / 1.094 [1.041, 1.174] | 0.561 [0.527, 0.591] / 3.909 [3.347, 4.271] |
| larceny | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| larceny | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| motor_vehicle_theft | model | 11 | 12878/13542/12878 | 0.166 [0.145, 0.187] / 21.792 [5.870, inf] | 0.585 [0.550, 0.625] / 1.699 [1.560, 1.843] | 0.755 [0.697, 0.818] / 2.139 [1.746, 2.421] |
| motor_vehicle_theft | direct | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |
| motor_vehicle_theft | benchmark | 0 | 0/0/0 | NA / NA | NA / NA | NA / NA |

### Protocol

Spatial folds remove one admitted feed city from direct allocation, residual training, and uncertainty training. Held-out pooled 2018–2024 incidents supply truth. Temporal folds truncate all feeds at 2021 or 2023 and score later incidents. The model fold uses the frozen E4 base cells; mixture weights v3, soft shrinkage, the rape triple, exposure ensemble weights, murder K, and tau are named in `reuse_flags` when reused.

Metrics are TVD, Spearman correlation, top-decile capture, exact spatial Wasserstein skill against the population null, and interval coverage. Confidence intervals resample cities. Negative skill and losses remain in the output.

### Not validated

- Allocation outside the 13 feed cities: about 95% of block groups.
- 98.1% of the edition's block groups outside the jurisdiction-panel training hull.
- Direct and benchmark interval calibration under spatial holdout.
- Spatial rape performance beyond one eligible feed city.

