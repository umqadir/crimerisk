# CrimeRisk methodology

Edition 2025.1. Data year 2025. History window 2018 to 2025.
Coverage is the 48 contiguous states and the District of Columbia.
Boundaries are 2020 Census block groups and tracts.

## Estimand

For each published area and each of the seven FBI Part I offenses, the estimand is
the number of offenses reported to and recorded by police in the data year.

```
expected_count_o   estimated offenses of type o in the area, in the data year
rate_o             100,000 * expected_count_o / denominator_o
index_o            100 * rate_o / national_rate_o
```

Every published rate and index is recomputable from a published count and a published
denominator. Expected counts are model estimates, not incident records.
An index describes an area. It is not a person's probability of victimization.

Two measures are published. Crime exposure uses the offense-specific activity or
opportunity denominator. Per resident uses resident population.

## Data sources

| Source | Vintage | Supplies |
|---|---|---|
| FBI UCR Return A master file | 2025 | Monthly agency offense counts, summary-family fallback |
| FBI CIUS local publication | 2025 | Annual agency and jurisdiction offense totals, first source priority |
| FBI NIBRS | 2018 to 2025 | SRS-equivalent annual agency rollups |
| FBI SRS | 2018 to 2025 | Annual agency totals where CIUS is unmatched |
| FBI Crime Data Explorer state estimates | 1979 to 2024 | State benchmark totals and their revision panel |
| State publications: FDLE, MS TOPS, NY DCJS, CT DESPP | 2024 | State-official annual rows in bounded state lanes |
| City open-data incident feeds, 13 jurisdictions | 2018 to 2025 | Geocoded within-jurisdiction offense shares |
| Census TIGER/Line | 2020 | Block, block group, tract and place boundaries |
| Decennial Census | 2020 | Block population for police-service footprint ownership |
| Census Population Estimates | Vintage 2025 | Resident population control |
| ACS 5-year | 2020 to 2024 | Household and socioeconomic covariates |
| LEHD LODES | 2023 | Workplace jobs and commuting flows |
| BLS QCEW | 2024 | County employment scaling |
| LandScan USA | 2021 | Ambient daytime population |
| Overture Maps places | 2026-06-17.0 | Premises, destinations and commercial activity counts |
| NLCD | 2023 | Land cover and impervious surface |
| Police-service registries: PA PCCD, MD, tribal, state-police posts | 2025 | Service footprints and provider identities |

Licenses and required source notices are in `ATTRIBUTION.md`.

## Jurisdiction totals

One official total per jurisdiction per offense per year, called the level lane.
Hard evidence admits or rejects a row outright: coverage contradictions, roster
impossibilities, identity failures, missing offense categories, official publications
that contradict the row. Soft signals hold a row for review and never drive automatic
upward repair: a drop below 0.25 of own history, an implausibly low Part I rate,
offense-mix anomalies, spikes, external-benchmark disagreement. A failed coverage
basis invalidates the vector. A failed offense invalidates only itself. Source
priority is `cius_publication_annual`, `local_publication_annual`,
`state_publication_annual`, `srs_return_a_annual`, `nibrs_srs_equivalent_annual`.

| Rung | `level_repair_mode` | Applied when |
|---|---|---|
| 1 | `none` | The row is admitted as a valid complete year |
| 2 | `source_supported_replacement`, `swap_reclassification` | A reviewed source contradicts the filed category |
| 3 | `partial_missing_increment` | Months are missing; observed mass locks as a lower bound |
| 4 | `covered_by_covering_agency` | Another agency reports this territory |
| 5 | `decayed_own_history` | The agency is silent and has clean continuous history |
| 6 | `pooled_silent_unit` | The agency is silent without usable own history |
| 7 | `coverage_adjusted_remainder` | A state non-municipal remainder lane is undercovered |
| 8 | `soft_benchmark_reconciliation` | Offense-level state totals disagree with the CDE estimate |
| 9 | `locked_structural_zero` | The zero is corroborated and locked |
| 10 | `unchanged_review_hold` | The row is held for review and left unchanged |

## Smoothing and national anchors

Temporal kernels smooth the annual panel: exponential weighting, one-year half-life
for murder, burglary, larceny and motor vehicle theft, two-year half-life for rape,
robbery and aggravated assault. Peer keys are offense by state by jurisdiction class
by population band, with shrinkage constant 0.5. One uniform offense-wide factor
reconciles local estimates to each national anchor. There is no state, county,
jurisdiction or map-edge calibration.

| Offense | National annual anchor |
|---|---:|
| Murder | 16,353.02 |
| Rape | 129,237.35 |
| Robbery | 205,633.29 |
| Aggravated assault | 857,367.99 |
| Burglary | 744,967.01 |
| Larceny | 4,160,052.33 |
| Motor vehicle theft | 795,522.22 |

## Police-service footprints

Each jurisdiction total is placed inside the territory its provider patrols.
Ownership is exclusive and resolved at Census-block support. Concurrent overlap
agencies are carried separately from primary patrol ownership. Overlap mass placed
on a custom footprint is raked to that agency's own ledger control. Mass that cannot
be placed goes to the unlocated-mass table.

## Neighborhood allocation

Each jurisdiction total is distributed across the block groups inside its footprint.
Block-group counts plus unlocated mass reconcile to the jurisdiction control.

Feed cities. Thirteen jurisdictions publish geocoded incident records that reconcile
to their own official totals. Their pooled 2018 to 2025 incident shares become the
posterior share evidence, blended with the model prior by a feed-quality weight.
Feeds are share evidence only. A feed never becomes a total.

Modeled areas. Everywhere else the share comes from one covariate model fitted on the
feed cities and applied nationally, then raked to the jurisdiction control.

| Covariate class | Used in allocation | Contents |
|---|---|---|
| `activity_exposure` | Yes | LODES jobs, commuting, Overture places |
| `roads_transport` | Yes | Road length by class, transit and fixed-guideway proximity |
| `land_cover` | Yes | NLCD cover and impervious share, land area |
| `institutional_anchors` | Yes | School, postsecondary and hospital density and proximity |
| `socioeconomic_household` | Under `proxy_review` only | Household and tenure composition that transfers within cities |
| Protected-class composition | No | Race, ethnicity, ancestry, language, nativity, sex, familial status, national origin |
| `between_only` features | No | Features that transfer between jurisdictions but not within them |

Direct protected-class fields are excluded regardless of predictive power. A feature
that helps between jurisdictions but not within them is excluded from allocation.
The release validator fails closed if a `between_only` or `excluded_protected`
feature is selected into the within-city residual allocator.

Murder and rape. One tract estimator serves both feed and modeled areas. Murder uses
a 10-year incident half-life and a 100-incident tract prior. Rape uses tract-level
information shrinkage. There is no post-model spatial smoothing and no cross-border
feathering.

## Denominators

The person-exposure denominator is a fixed convex mix of three legs: a residential
leg, a LandScan daytime leg and a LODES daytime-jobs proxy. The residential leg in
2025.1 is **LandScan nighttime population**, not Census residential population. The
per-offense leg weights are read from `configs/exposure_ensemble_weights_v1.csv` and
are never re-derived at build time; each normalizer is then rescaled so its total over
the published universe equals resident population (339,614,776).

| Offense | Crime exposure denominator | Residential leg | Per resident denominator |
|---|---|---|---|
| Murder | Person exposure ensemble, 0.9 residential / 0.0 LandScan day / 0.1 jobs | LandScan nighttime | Resident population |
| Rape | Person exposure ensemble, 0.7 / 0.1 / 0.2 | LandScan nighttime | Resident population |
| Robbery | Person exposure ensemble, 0.7 / 0.3 / 0.0 | LandScan nighttime | Resident population |
| Aggravated assault | Person exposure ensemble, 0.7 / 0.1 / 0.2 | LandScan nighttime | Resident population |
| Burglary | Residential and weighted commercial premises | Not used | Resident population |
| Larceny | 0.5 person ensemble (0.4 / 0.3 / 0.3) + 0.3 destination POI + 0.2 retail jobs | LandScan nighttime | Resident population |
| Motor vehicle theft | Household and commuter vehicle exposure | Not used | Resident population |

The person-exposure denominator uses modeled daytime population where it exceeds
resident population. It is not a measured visitor count.

A Census-residential arm exists in the code
(`build-exposure-normalizers --residential-leg-source census_release_population`),
which replaces the nighttime leg with release Census population nationwide and writes
a separate `_census_residential` artifact under its own normalizer version. It is a
sensitivity arm and is not what 2025.1 publishes; the shipped artifact records
`residential_leg.source = landscan_night`. Murder and rape leg weights are marked
provisional in the weight table: the evidence supports the direction, not the exact
triple.

## Composites

Three composites are published on each measure: Overall over all seven offenses,
Violent over murder, rape, robbery and aggravated assault, Property over burglary,
larceny and motor vehicle theft.

```
index_composite = sum_o ( w_o * index_o )
w_o             = national expected count of offense o / national expected count of the set
```

The exposure composite averages the exposure indexes. The resident composite averages
the resident indexes over a shared denominator, count first.

Support recipe by measure:

| Measure | Support | Murder and rape term | Withheld when |
|---|---|---|---|
| Crime exposure | Tract | The tract's own index | Any of the seven component indexes is absent |
| Crime exposure | Block group | The parent tract's index | Any of the five volume-offense indexes is absent, or the parent tract has no murder or rape index |
| Per resident | Tract | The tract's own count | Any of the seven components is unpublishable |
| Per resident | Block group | The block group's own count; murder and rape do not gate | Any of the five volume offenses is unpublishable |

A suppressed volume-offense index is not filled from the parent tract. A block-group
exposure composite is recomputable from the block-group row (five volume offenses) and
the parent tract row (murder and rape).

No harm-weighted composite is published. Public composite names and their internal
fields:

| Public | Crime exposure field | Per resident field |
|---|---|---|
| Overall | `multi_offense_relative_score_event_weighted` | `index_event_burden_resident` |
| Violent | `multi_offense_relative_score_personal_event_weighted` | `index_personal_burden_resident` |
| Property | `multi_offense_relative_score_property_event_weighted` | `index_property_burden_resident` |

## Support rules

| Item | Rule |
|---|---|
| Robbery, aggravated assault, burglary, larceny, motor vehicle theft | Published at block group and tract |
| Murder and rape | Published at tract only |
| Murder and rape at block group | Expected count only, for reconciliation; rate and index are null |
| Composites at block group | Exposure: block-group indexes for the five volume offenses, the parent tract's index for murder and rape. Per resident: block-group counts, with murder and rape not gating |
| County display rollup | At least 2 tracts and at least 2,500 residents |
| Jurisdiction and state comparison values | At least 2 block groups and at least 2,500 residents |
| Map layers | County z3 to z7, tract z5 to z12, block group z8 to z12 |

## Suppression rules

| Rule | Value |
|---|---|
| Resident-measure population floor | 50 residents |
| Exposure denominator floor when resident population is below 50 | 500 exposure units |
| Offense index below its exposure floor | Null, `estimate_mode = insufficient_exposure` |
| Composite with any suppressed component | Null |
| Suppressed rare-offense composite term | No parent-tract fill |
| Corroborated structural zero | Published at zero |
| Special-use type | Annotation only; it suppresses nothing by itself |
| Incident-level records | Never published |

## Uncertainty

The build carries a 100-draw uncertainty layer internally, but held-out coverage of
those intervals is below nominal at every level (50, 80 and 95; see the interval
calibration table in `docs/EVALUATION.md`), so 2025.1 publishes no interval and no
reliability tier. The per-cell p10, p90 and `reliability_tier` fields are not in the
download, the lookup shards or the map.

## Known limitations

- Only crime reported to and recorded by police is represented.
- Offense classification and reporting completeness vary by agency and by state.
- Neighborhood values are modeled shares of agency totals, not counted incidents.
- Allocation outside the 13 feed cities is modeled, covering about 95% of block groups.
- 233,584 of the edition's 238,193 block groups, 98.1%, sit outside the jurisdiction-panel training hull on at least one governed covariate (`extrapolation_flag` in `state/modeling/bg_mixture_experts_2025.parquet`).
- Held-out evaluation is possible only in feed cities. Feed-city performance is not a bound on performance elsewhere.
- Spatial held-out TVD: the population null is lower on motor vehicle theft (0.287 against 0.298) and the primary-exposure null on burglary (0.280 against 0.287). Temporal rows where past counts are lower are listed in `docs/EVALUATION.md`.
- Agencies that filed no 2025 row receive an imputed total. Residents under a control imputed from a pooled peer unit (`level_repair_mode = pooled_silent_unit`) are 2.08% of the published population on larceny and 2.00% to 2.15% across the seven offenses; including controls carried from the agency's own clean history (`decayed_own_history_or_pooled`) the range is 3.88% to 5.86%.
- Murder and rape have no block-group rate or index.
- Exposure denominators do not represent tourist and event populations.
- Sparse small-area counts carry wide internal intervals, which 2025.1 does not publish.
- 2020 boundaries do not reflect later boundary changes.
- Alaska, Hawaii and the territories are excluded.
- An index is not an individual person's probability of victimization.
- Los Angeles has no admitted direct-incident allocation feed in this edition.

## Reproduction

- Build steps and commands: `docs/BUILD.md`
- Published columns: `docs/FIELDS.md`
- Evaluation protocol and results: `docs/EVALUATION.md`
- Level lane: `src/crimerisk/level_lane.py`, `src/crimerisk/controls.py`, `src/crimerisk/smoothed_controls.py`
- Allocation: `src/crimerisk/allocation.py`, `src/crimerisk/mixture_allocation.py`
- Denominators and composites: `src/crimerisk/denominators.py`, `src/crimerisk/composites.py`
- Footprints: `configs/overlap_custom_footprints.csv`, `configs/pa_police_service_coverage.csv`
- Feed admissions: `configs/city_incident_priority.csv`, `configs/city_incident_sources.csv`
- Release validator: `scripts/diagnostics/validate_release_outputs.py`
- Public schema: `frontend/build/crschema.py`

## Reference

- Agency is the atomic crime-reporting unit. Jurisdiction is the atomic scaling unit.
- The state non-municipal remainder is a control layer, not a balancing residual.
- FBI estimated state totals are aggregates of reported agency values plus FBI agency-level estimation.
- `FBI estimated total minus internal total` is not residual geography.
- Overlap agencies are resolved by subtype: absorbed, modeled on their own footprint, or kept as a coarse layer.
- Geometry is built from 2020 blocks upward, not from tract centroids downward.
- AGS CrimeRisk methodology files in `docs/` are a reference for target mechanics, not a numeric input.
- FBI record layouts and offense definitions: `docs/archive/2026-09/FBI-DATA-GUIDE.md` and `docs/archive/2026-09/ucrbook_full.md`.
