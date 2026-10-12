# CrimeRisk methodology

Edition 2025.3. Data year 2025. History window 2018 to 2025.
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
opportunity denominator. Per resident uses resident population. Crime exposure is the
default measure; the map card shows both values.

## What each index means

One or two sentences per published index, as shown on the map card and the methods page.

| Index | Description |
|---|---|
| Robbery, crime exposure | Relative risk of robbery for the people present (residents and daytime population; the jobs term has weight 0), 100 = U.S. average. The level combines the police agency's 2025 robbery count, where it filed a complete year, with a forecast from its earlier years (two-year half-life), each weighted by its precision, and is matched to the 2025 state total (the FBI's state estimate plus documented agency corrections); the share within the agency comes from mapped police incidents in the cities that publish them and from a statistical model elsewhere. |
| Aggravated assault, crime exposure | Relative risk of aggravated assault for the people present (residents, daytime population and workers), 100 = U.S. average. Level and share as for robbery (two-year half-life). |
| Larceny/theft, crime exposure | Relative risk of larceny/theft for the people and places here: half people present, 30% destination places, 20% retail jobs. Level and share as for robbery (one-year half-life). |
| Burglary, crime exposure | Relative risk of burglary per premises: households, group-quarters residents (0.265 premises each), destination places and retail and industrial jobs, weighted by a fit to mapped burglaries. Level and share as for robbery (one-year half-life). |
| Motor vehicle theft, crime exposure | Relative risk of vehicle theft per vehicle present: household vehicles plus commuters' cars, never fewer than 0.25 vehicles per person present. Level and share as for robbery (one-year half-life). |
| Murder, crime exposure | A smoothed census-tract estimate per person present: the agency level combines its complete 2025 count with a forecast from earlier years (one-year half-life) that leans on similar agencies when the agency is small, each weighted by its precision, and the tract share comes from the city's own murders (10-year half-life) anchored to its violent-incident pattern (feed cities) or from the model elsewhere. It shows where murder risk is concentrated; in a small town one murder can move it several-fold. |
| Rape, crime exposure | A smoothed census-tract estimate per person present: the agency level combines its complete 2025 count with a forecast from earlier years (two-year half-life), each weighted by its precision, and the tract share comes from the model nearly everywhere. Rape reporting and definitions vary across agencies more than for other offenses. |
| Overall, Violent, Property, crime exposure | The component exposure indexes averaged with weights equal to each offense's share of U.S. offenses in the set (Overall: larceny 61.2%, aggravated assault 12.9%, motor vehicle theft 10.7%, burglary 10.4%, robbery 2.7%, rape 1.9%, murder 0.2%; Violent: aggravated assault 72.9%, robbery 15.0%, rape 10.9%, murder 1.3%; Property: larceny 74.4%, motor vehicle theft 13.0%, burglary 12.6%). |
| Any offense, per resident | The same expected offenses divided by residents, relative to the U.S. Visitor and commercial areas with few residents score high. |
| Composites, per resident | The set's expected offenses added up and divided by residents, relative to the U.S. |

Composite weights: national sums of `expected_count_<offense>` in
`state/candidates/rc3-2025.3/crimerisk_block_group_2025_ags_core.parquet`. Group-quarters
weight: `burglary_group_quarters_exposure_weight` in the same file (0.265311).

## Data sources

| Source | Vintage | Supplies |
|---|---|---|
| FBI UCR Return A master file | 2025 | Monthly agency offense counts, summary-family fallback, per-offense monthly profile for the partial-year screen |
| FBI CIUS local publication | 2025 | Annual agency and jurisdiction offense totals, first source priority |
| FBI NIBRS | 2018 to 2025 | SRS-equivalent annual agency rollups |
| FBI NIBRS published agency tables | 2024, 2025 | Agency offense tables |
| FBI SRS | 2018 to 2025 | Annual agency totals where CIUS is unmatched |
| FBI NIBRS batch header | 1991 to 2024 | `covered_by_ori` service relationships |
| FBI Crime Data Explorer state estimates | 1979 to 2025 | State targets, revision panel, state levels for the forecast |
| State publications: FDLE, MS TOPS | 2024 to 2025 | State-official annual rows in bounded state lanes; rape not admitted (definition mismatch) |
| State publication: NY DCJS | 2021 to 2025 (2025 from the DCJS Tableau table, retrieved 2026-10-11) | State-official annual rows |
| State publication: CT DESPP | 2021 to 2025 | Town rows for the Connecticut town lane |
| City open-data incident feeds, 15 jurisdictions | 2018 to 2025; St. Louis and Denver 2021 to 2025, Minneapolis and Boston 2019 to 2025, Houston 2019 to 2025, Mesa 2020 to 2025, Atlanta 2018 and 2021 to 2025, New York murder 2020 to 2025; Philadelphia robbery 2018, 2020 to 2022 and 2025, larceny 2018 and 2020 to 2025, motor vehicle theft 2022 to 2025, murder 2018 and 2020 to 2022 | Geocoded within-jurisdiction offense shares |
| Census TIGER/Line | 2020 | Block, block group, tract, place and AIANNH/AITSN boundaries |
| BIA AIAN National Land Area Representation | Live layer, retrieved 2026-10-11 | Tribal trust and reservation land for tribal territory footprints |
| Census TIGER/Line roads | 2025 | Road length by class (structural cache) |
| Decennial Census | 2020 | Block population for police-service footprint ownership; DHC group-quarters residents by block group |
| Census Population Estimates | Vintage 2025 | Resident population control |
| ACS 5-year | 2020 to 2024 | Household and socioeconomic covariates |
| LEHD LODES | 2023 | Workplace jobs and commuting flows; WAC and RAC block files for tribal territory shares |
| LandScan USA | 2021 | Ambient daytime population |
| Overture Maps places | 2026-09-23.1 | Premises, destinations and commercial activity counts |
| NLCD | 2025 | Land cover and impervious surface |
| FHWA HPMS; National Transit Map stops | 2024; NTAD layer last edited 2026-04-21 | Road and transit covariates (structural cache) |
| Police-service registries: PA PCCD, MD, tribal, state-police posts | 2025 | Service footprints and provider identities |

Every input is at its latest available vintage as checked on 2026-10-06 (NY DCJS 2025, the
BIA LAR layer and the Atlanta and Houston feeds as retrieved 2026-10-10 to 2026-10-11).
Documented exceptions:

| Input | Status |
|---|---|
| National Transit Map stops | The NTAD layer (last edited 2026-04-21) is the only vintage available; no dated history exists, so every year's transit covariates use it. The snapshot used in 2025.2 is not re-downloadable |
| Restricted agencies | 27 agencies under 31 reporting-regime restrictions (`configs/reporting_regime_restrictions.csv`): their filings for the restricted offenses, from the restriction's first year, are not admitted |
| Houston feed | Stage-7 cache of the HPD NIBRSPublicView files pulled 2026-10-11; the files served on recheck are identical to the cache |
| Mesa feed | Frozen 2020 to 2025 cache; the release endpoint is gone |
| TIGER places and county subdivisions | 2020; TIGER 2025 not migrated |
| LandScan USA, ACS, LODES | Latest public vintages: 2021, 2020 to 2024, 2023 |

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

Adjudicated reporting history (`src/crimerisk/reporting_history.py`,
`state/controls/reporting_history_2025.parquet`):

| Item | Rule |
|---|---|
| History years | Each year from 2018 runs the target-year admission sequence in order (level-lane admission, masked-gap detector, stage-1 directives), seeing only years up to it |
| Refused year | Missing (`usable_as_observed` False), not zero; filed value kept in `history_original_count` |
| Replacement | Admitted at the replacement count |
| Consumers | Level repair, risk signals, jurisdiction-year panel, smoothing, mixture training and the level holdout read the one history |
| 2018 to 2024 rows | 589,074 admitted complete, 96,501 admitted partial (lower bound), 91 admitted at a replacement count, 97,401 rejected, 146,281 not reported |
| Reporting-regime restriction | `configs/reporting_regime_restrictions.csv`: from `effective_from_year` until `lifted_from_year`; year-keyed hard evidence wins for its own year |
| Clean jurisdiction-year | Complete admitted twelve-month filings carry at least 0.95 of the expected mass; expected mass of a missing agency is the median of its complete years in the three years before; an agency that filed before with no complete earlier year makes the year not clean |
| Partial-year screen | Masked-gap `partial_months` signature reads the NIBRS lane and, where no NIBRS row exists, the Return A master's per-offense monthly profile |
| FDLE months | A month counts as reported only where the agency's 37-offense total is above zero |
| Semantic admission | A source whose definition differs for an offense is not admitted for it: FDLE "Forcible Sex Offenses" and MS TOPS "Non-Consensual Sex Offenses" (include fondling) are not rape; NY DCJS "Rape" and CT DESPP are admitted |
| NY State Police | DCJS county State Police rows match the FBI "SP: <county> COUNTY" ORI (55 of 57 counties) |

| Rule | Value |
|---|---|
| State-publication agency-year below 0.25 x the full-year FBI report for the same agency-year | Not a candidate lane |
| Structural-zero lock | Target year's own row is a filed Part I vector of exactly 0 on at least 11 months |
| Partial-year floor | Partial offenses compared with their own history; only the partial rows are refused |
| Connecticut State Police (CTCSP0000) | Residualized on each resident-trooper town's report or fill; town lane 2021 to 2025 |
| Agency whose every territory a service relationship gives to another provider | No 2025 numerator and no fill estimate of its own |
| Municipal jurisdiction with a zero control on every offense | Takes its covering pool's rate by exposure: contracting agency, else county nonmunicipal pools, else a PSP service unit |
| Miami-Dade Sheriff's Office | Matched to the former Miami-Dade PD, `FL0130000` |

| Rung | `level_repair_mode` | Applied when |
|---|---|---|
| 1 | `none` | The row is admitted as a valid complete year |
| 2 | `source_supported_replacement`, `swap_reclassification` | A reviewed source contradicts the filed category |
| 3 | `partial_missing_increment` | Months are missing; observed mass locks as a lower bound |
| 4 | `covered_by_covering_agency` | Another agency reports this territory |
| 5 | `decayed_own_history` | The agency is silent or restricted and has clean continuous history |
| 6 | `pooled_silent_unit` | The agency is silent without usable own history |
| 7 | `coverage_adjusted_remainder` | A state non-municipal remainder lane is undercovered |
| 8 | `soft_benchmark_reconciliation` | Offense-level state totals disagree with the CDE estimate |
| 9 | `locked_structural_zero` | The zero is corroborated and locked |
| 10 | `unchanged_review_hold` | The row is held for review and left unchanged |

## Smoothing and state totals

Each jurisdiction-offense level is the clean 2025 count weighted by its precision against a
past-only forecast (`smoothed_controls.precision_weighted_levels`):

```
level = w c + (1 - w) H
w     = V / (V + H)
V     = tau2 H^2 + var(H)      unit with clean history
V     = tau2_peer H^2          peer-only forecast
```

Without a clean 2025 count the level is H.

| Item | Rule |
|---|---|
| Forecast H | Exponential kernel over the clean years before 2025, each year moved to the 2025 state level (c_t x S_2025 / S_t, CDE state estimates); half-life one year for murder, burglary, larceny and motor vehicle theft, two years for rape, robbery and aggravated assault; shrunk (k = 0.5) to the leave-one-out past-only peer forecast (state x class x population band, else national class x band); murder: information posterior, k = 10,000 |
| tau2, tau2_peer | Method of moments over earlier unit-years, finest of offense x band, offense, all offenses with at least 30 pairs; otherwise `fallback_one_more_year` |
| var(H) | Poisson variance of the scaled forecast, s^2 H sum(w^2 f) / (sum w)^2 |
| Current-year check | Review flag only (`current_year_check`): 25k+ residents, count below half the forecast and half the peer ratio, negative-binomial tail below 1/1,000; listed in the validation summary; changes no control. 2025: 10 flagged cells unconfirmed (Known limitations), 3 flagged cells confirmed by `configs/current_year_confirmations.csv` (23 rows) |
| Diagnostics | Per unit: H, var(H), forecast years, c, w, tau2 and its level and support, in `state/controls/jurisdiction_controls_smoothed_2025.parquet` |

The smoothed levels are raked within each state: one uniform factor per state and offense,
so each state-offense total equals its state target. The national total is the sum of the
state targets. There is no county, jurisdiction or map-edge calibration.

| Item | Rule |
|---|---|
| State target | FBI CDE 2025 state estimate plus, for each agency with a documented correction whose own twelve-month report the FBI used (all FBI sources agreeing on the count), corrected count minus FBI count |
| FBI imputations | Published FBI imputations for partial reporters and non-reporters held fixed (correction 0) |
| Correction basis | `measured_external_replacement` or `model_repair_of_documented_defect` (`state/controls/state_anchor_contributions_2025.csv`) |
| Corrections, 2025 | 59 included rows, 13 agencies, 10 states: 12 measured replacements (net +17,548), 47 model repairs (net +10,741); 71 excluded (35 FBI-imputed partial reporters, 36 FBI-imputed non-reporters) |
| Sensitivity | Target without model-derived agency deltas, recorded exception retained (`state_anchor_count_external_only`, `state/controls/state_anchor_sensitivity_2025.csv`) |
| Recorded exception | Accounting total for a documented CDE break, with effective years (`configs/state_anchor_exceptions.csv`): Florida motor vehicle theft, 2025 only (FBI file 10,555 against FDLE 18,819 on the same 215 agencies) |
| State-offense cells | 343: 294 CDE, 48 CDE plus corrections, 1 recorded exception |

Source: `state_anchor_source`, `state_anchor_count` in
`state/controls/jurisdiction_controls_smoothed_2025.parquet`.

| Offense | National total, 2025 (sum of state targets) |
|---|---:|
| Murder | 14,238.92 |
| Rape | 122,411.33 |
| Robbery | 168,459.46 |
| Aggravated assault | 820,815.68 |
| Burglary | 659,230.41 |
| Larceny | 3,894,469.98 |
| Motor vehicle theft | 681,114.13 |

Block-group expected counts plus unlocated mass equal these totals for every offense.

## Police-service footprints

Each jurisdiction total is placed inside the territory its provider patrols.
Ownership is exclusive and resolved at Census-block support. Concurrent overlap
agencies are carried separately from primary patrol ownership. Overlap mass placed
on a custom footprint is raked to that agency's own ledger control. A state-police unit
whose footprint holds fewer residents than the agency's FBI population served keeps
only that share of its total on the footprint; the rest is spread over its county in
the county's allocation shape. Mass that cannot be placed goes to the unlocated-mass
table.

Service relationships (`src/crimerisk/service_relationships.py`): effective-dated, first
match per territory and year from `configs/service_relationships.csv` (34 rows), the
consolidated-footprint registry, then the FBI NIBRS batch header's `covered_by_ori`.

| Relation | Territory |
|---|---|
| `contract`, `consolidated` | Joins the provider's municipal jurisdiction; with a county or state provider, joins the state remainder whose county pools carry that provider's total |
| `county_unincorporated` | No geometry change; the county remainder counts toward the provider's footprint |
| `exclusive`, `supplemental` | No change |
| No row | The municipality keeps its own geometry; a non-reporting department's control comes from the silent-unit imputation |

| Item | Value |
|---|---|
| Territory check | Agencies serving 10,000 or more: FBI 2022 population against the resolved footprint (city territory after relationship moves, county remainder of each `county_unincorporated` row, custom footprints), grown 2020 to 2022; outside [1/1.3, 1.3] fails unless a recorded exception covers it (`configs/service_territory_exceptions.csv`) |
| Territory check, 2025 | 3,496 within band, 16 recorded exceptions, 0 unresolved (`state/controls/service_territory_check_2025.csv`) |
| Municipal jurisdictions with block groups | 12,104 |
| County remainder partition | Ownership first; only owned counties take part; evidence = risk signal plus imputed silent mass; Dirichlet prior strength k = 0.5 over owned exposure (`county_remainder_partition_2025.parquet` in the candidate) |

Tribal territory footprints (`scripts/build/01_tribal_territory_footprints.py`,
`src/crimerisk/territory_shares.py`; 3,635 rows, 158 agencies, `weight_share_basis =
territory_prior` in `configs/overlap_custom_footprints.csv`):

| Item | Rule |
|---|---|
| Tribal territory | TIGER 2020 AIANNH/AITSN area plus mapped BIA AIAN National LAR land; unpopulated LAR blocks by majority area; facility blocks with a destination point on LAR land; populated LAR-only blocks excluded; facilities served by another agency under a recorded agreement excluded (`configs/tribal_facility_service_agreements.csv`: Ko-Kwel Casino, North Bend OR) |
| Tribal footprint | Every 2020 block group with a territory block, whatever its population |
| Territory share s | Per offense: legs (residents, LandScan day, daytime jobs, destination POIs, retail, industrial and WAC jobs, housing units, land) on the territory part over the block group's, weighted by their contribution to the block group's denominator; land only where a block group has no denominator leg |
| Allocation weight | s (prior x s) + (1 - s)(r x offense exposure on the territory part); r = footprint prior-implied rate; s = 1 gives prior x s; normalised per agency and offense |
| Covered population | `configs/overlap_custom_footprint_resident_coverage.csv`, unchanged; `territory_share_residents` equals it |

## Neighborhood allocation

Each jurisdiction total is distributed across the block groups inside its footprint.
Block-group counts plus unlocated mass reconcile to the jurisdiction control.

Feed cities. Fifteen jurisdictions publish admitted geocoded incident records: Atlanta,
Austin, Baltimore, Boston, Chicago, Denver, Houston, Mesa, Minneapolis, New York,
Philadelphia, San Francisco, Seattle, St. Louis, Washington. Their pooled incident shares
become the posterior share evidence, blended with the model share. Feeds are share
evidence only. A feed never becomes a total. Atlanta and Houston contracts count one row
per incident and offense (`mapping.dedup_unit: incident_offense`); rape is truth-only in
both.

```
delta          = 1 - exp(-|log(F / C)|)
prior_fraction = min(0.995, 1 - (1 - delta) * match_rate * F / (F + 25))
F              feed count for the city and offense, 2025
C              the jurisdiction's smoothed, state-raked control
```

Modeled areas. Everywhere else the share comes from a three-expert mixture, then a
residual model, then a rake to the jurisdiction control.

| Component | Specification |
|---|---|
| Expert b, model-surface prior (stage 10b) | Per offense, mean of 10 HistGradientBoosting fits, fit k on training jurisdictions whose sha256(seed, k, jurisdiction) falls below 0.8, seed 20260926; depth 3, min leaf 100, l2 10, 500 iterations, learning rate 0.03; trained on FBI jurisdiction totals (jurisdictions of at least 10,000 residents) over the governed block-group covariates; 178 features (burglary 138) |
| Expert g, GBM expert (stage 11) | Bounded GBM on FBI jurisdiction rates over the stage-10b feature catalog; predictions outside the learned intensity envelope (0.5th to 99.5th percentile) are soft-shrunk toward it (see Envelope treatment) |
| Expert e, exposure | The offense's opportunity normalizer |
| Mixture weights | Per offense by jurisdiction-size stratum, `configs/mixture_ship_weights_v3.csv`, selected on the 20-source E4 corpus; read, never re-derived |
| Residual model (stage 11b) | Target log(true share / prior share); mean of 10 HistGradientBoosting fits on keyed 80% city subsets, seed 20260927; depth 5, min leaf 20, l2 1, 500 iterations, learning rate 0.03; log-ratio clip 8; trained on 33 cities (feed and validation cities); 229 features |
| Residual transfer tau | 1 for robbery, aggravated assault, larceny, motor vehicle theft; 0 for burglary, murder, rape |
| Envelope treatment (`soft_shrinkage` in the candidate manifest) | Replaces the hard clip to the GBM intensity envelope and the hard model-only rate-ratio caps. Inside a bound the value is unchanged; outside it the log excess is compressed, r' = bound x exp(s x nu x log1p(abs(log(r / bound)) / nu)), nu = 2. GBM envelope: s = 1, or 0.25 for block groups outside the training hull (extrapolation weight 0.75). Model-only rate-ratio bounds against the jurisdiction median, s = 1: murder 57 (tract), robbery 22 (block group); the removed mass is redistributed within the source jurisdiction and offense |
| Larceny finished-share blend | 0.8 model, 0.1 population, 0.1 primary exposure on model-only jurisdictions (`configs/finished_share_blend_v1.csv`); other offenses 1, 0, 0 |

Feature policy: `state/modeling/feature_transfer_policy_2025.parquet`, 342 columns.

| Class | Used in allocation | Columns | Feature groups |
|---|---|---:|---|
| `between_and_within` | Yes | 160 | `activity_exposure_commercial` (67), `lodes_industry_activity` (20), `roads_transport` (33), `land_cover` (30), `population_density` (10) |
| `proxy_review` | Yes | 69 | `income_wealth` (9), `poverty_snap` (2), `education_attainment` (5), `employment_labor` (4), `lodes_earnings` (3), `lodes_education` (4), `housing_tenure_overcrowding` (26), `rent_burden` (4), `digital_access_proxy` (12) |
| `between_only` | No | 86 | `commute_transport_exposure` (11), `vehicle_availability` (14), `state_fixed_effect` (49), `household_composition_other` (4), `residential_mobility_domestic` (4), `lodes_age` (3), `other_covariate` (1) |
| `excluded_protected` | No | 27 | `sex_composition` (13), `familial_status_composition` (13), `national_origin_proxy` (1) |
| Race, ethnicity, ancestry, language | No | 0 | Not in the feature set |

The release validator fails closed if a `between_only` or `excluded_protected`
feature is selected into the within-city residual allocator.

Murder and rape. Tract support in feed and modeled areas.

| Item | Murder | Rape |
|---|---|---|
| Feed-city tract posterior | City murders, 10-year incident half-life; prior = the city's aggravated-assault and robbery tract share, strength chosen by rolling origin from {0, 10^(k/4), k = 0..32}; fallback strength 100 incidents | St. Louis: city rapes with the model share as prior, model weight 0.22 (prior strength 277.5 incidents); other feed cities: model share |
| Feed-city prior strength, 2025 | Seattle 56.2, Washington 562.3, Minneapolis 1,000, St. Louis 1,000, Philadelphia 1,778.3, Chicago 3,162.3, Atlanta 10,000, Houston 10,000, New York 10,000, Denver 10^8 | — |
| Modeled-area tract information constant K | 1 | 100 |
| Within-jurisdiction tract rate-ratio bound, model-only | 57, soft-shrunk (excess compressed, not truncated) | — |

There is no post-model spatial smoothing and no cross-border feathering.

Provenance (`source_mode_<offense>`, `source_mode_dominant_share_<offense>`;
`src/crimerisk/confidence.py`):

| Item | Rule |
|---|---|
| Direct part of a feed-city cell | (1 - prior_fraction) x direct share / posterior share of the cell's share; 0 outside an active city posterior |
| `direct_city_incident` | Direct part at least the modeled part, dominant share at least 0.6 |
| `modeled_transfer` | Modeled part larger, dominant share at least 0.6 |
| `mixed` | Dominant share below 0.6 |
| Card source line | Expected-count share from direct records: at least 0.9 "From local police incident records", at most 0.1 "Modeled from jurisdiction totals and regional patterns", otherwise "Mostly from local police incident records" or "Mostly modeled from jurisdiction totals" |
| Card jurisdiction line | `level_admission_status = valid_complete_year` and `level_repair_mode = none` on every offense: "Jurisdiction total: modeled from a complete 2025 police filing"; on none: "Jurisdiction total: modeled; the 2025 police filing was incomplete or missing"; otherwise "Jurisdiction total: modeled from the 2025 police filing, complete for some offenses only" |

## Denominators

The person-exposure denominator is a fixed convex mix of three legs: a residential
leg, a LandScan daytime leg and a LODES daytime-jobs proxy. The residential leg is
release Census population (`residential_leg.source = census_release_population`). The
per-offense leg weights are read from `configs/exposure_ensemble_weights_v1.csv` and
are never re-derived at build time.

| Offense | Crime exposure denominator | Residential leg | Rescaled to resident population | National total | Per resident denominator |
|---|---|---|---|---:|---|
| Murder | Person exposure ensemble, 0.9 residential / 0.0 LandScan day / 0.1 jobs | Census population | Yes | 339,614,776 | Resident population |
| Rape | Person exposure ensemble, 0.7 / 0.1 / 0.2 | Census population | Yes | 339,614,776 | Resident population |
| Robbery | Person exposure ensemble, 0.7 / 0.3 / 0.0 | Census population | Yes | 339,614,776 | Resident population |
| Aggravated assault | Person exposure ensemble, 0.7 / 0.1 / 0.2 | Census population | Yes | 339,614,776 | Resident population |
| Burglary | Households + 24.254 x destination POI + 0.322 x retail jobs + 0.044 x industrial jobs + 0.265 x group-quarters residents | Not used | No | 181,423,193 premises | Resident population |
| Larceny | 0.5 person ensemble (0.4 / 0.3 / 0.3) + 0.3 destination POI + 0.2 retail jobs | Census population | Yes | 339,614,776 | Resident population |
| Motor vehicle theft | max(household vehicles + LODES jobs x county auto-commute share, 0.2511 x daytime person presence) | Not used | No | 350,597,394 vehicles | Resident population |

| Parameter | Value |
|---|---|
| Burglary coefficients | Nonnegative least squares on feed-city burglary incidents, 2018 to 2025: 33 cities, 24,132 block groups, 94,566 rows |
| Industrial jobs | LODES manufacturing + wholesale + transport and warehousing jobs |
| Group-quarters residents | 2020 DHC `census2020_gq_total` |
| Daytime person presence | Residents plus net LODES commuters, raised to LandScan daytime population where larger, HQ-jobs capped |
| Vehicle floor ratio | 0.2511: presence-weighted 5th percentile of vehicles per person present |
| National reference | One per offense and measure: every expected count over every denominator unit on the surface, before any publication mask (cells whose denominator the build declares invalid excluded); the same scalar at block group and tract and for the composites (`allocation.national_reference_rate`) |

Source: `burglary_commercial_calibration` in
`state/candidates/rc3-2025.3/manifest.json`; `primary_denominator_<offense>` in the
block-group parquet. No denominator is a measured visitor count. Murder and rape leg
weights are marked provisional in the weight table.

## Composites

Three composites are published on each measure: Overall over all seven offenses,
Violent over murder, rape, robbery and aggravated assault, Property over burglary,
larceny and motor vehicle theft.

```
index_composite = sum_o ( w_o * index_o )
w_o             = national expected count of offense o / national expected count of the set
```

| Composite | Weights, 2025.3 |
|---|---|
| Overall | Larceny 0.612, aggravated assault 0.129, motor vehicle theft 0.107, burglary 0.104, robbery 0.027, rape 0.019, murder 0.002 |
| Violent | Aggravated assault 0.729, robbery 0.150, rape 0.109, murder 0.013 |
| Property | Larceny 0.744, motor vehicle theft 0.130, burglary 0.126 |

The exposure composite averages the exposure indexes. The resident composite averages
the resident indexes over a shared denominator, count first.

Support recipe by measure:

| Measure | Support | Murder and rape term | Withheld when |
|---|---|---|---|
| Crime exposure | Tract | The tract's own index | Any of the seven component indexes is absent |
| Crime exposure | Block group | The parent tract's index | Any of the five volume-offense indexes is absent, or the parent tract has no murder or rape index |
| Per resident | Tract | The tract's own count | Residents below the resident floor (300 ordinary, 550 other special-use types), or any of the seven component indexes is unpublishable |
| Per resident | Block group | The block group's own count; murder and rape do not gate | As for tract, over the five volume offenses |

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

One floor per special-use type, on both measures:

| `special_use_type` | Per-resident floor | Crime-exposure floor, murder, rape, robbery, aggravated assault |
|---|---:|---:|
| `ordinary` | 300 residents | 300 resident-equivalents |
| `park_open_space`, `industrial_employment`, `institutional_facility`, `campus_institution`, `transient_destination`, `group_quarters_other`, `unknown_special_use` | 550 residents | 550 resident-equivalents |

| Rule | Value |
|---|---|
| Resident-equivalents | Denominator x national residents / national denominator total |
| Crime-exposure floor, burglary | At least 10 premises; at least 500 where residents are below 50 |
| Crime-exposure floor, larceny | At least 50 person units; at least 500 where residents are below 50 |
| Crime-exposure floor, motor vehicle theft | At least 50 vehicles; at least 500 where residents are below 50 |
| Murder and rape crime exposure, additionally | At least 550 residents |
| Offense index below its exposure floor | Null, `estimate_mode = insufficient_exposure` |
| Map card where an index is withheld | Expected annual count, named by area type (`SUPPRESSED_COUNT_COPY` in `frontend/build/crschema.py`) |
| Composite with any suppressed component | Null |
| Suppressed rare-offense composite term | No parent-tract fill |
| Corroborated structural zero | Published at zero |
| Withholding a cell | Moves no other cell's index (one frozen national reference) |
| Incident-level records | Never published |

Published cells, `rc3-2025.3`:

| Field | Block groups | Tracts |
|---|---:|---:|
| Murder, crime exposure | — | 82,742 |
| Rape, crime exposure | — | 82,743 |
| Robbery, crime exposure | 235,020 | 83,076 |
| Aggravated assault, crime exposure | 234,673 | 83,039 |
| Burglary, crime exposure | 236,210 | 83,032 |
| Larceny, crime exposure | 236,733 | 83,170 |
| Motor vehicle theft, crime exposure | 236,701 | 83,169 |
| Overall, crime exposure | 234,002 | 82,741 |
| Violent, crime exposure | 234,185 | 82,742 |
| Property, crime exposure | 236,155 | 83,023 |
| Every per-resident index and composite | 233,612 | 82,797 |
| Universe | 238,193 | 83,776 |

## Uncertainty

The build carries a 100-draw uncertainty layer internally; its control dispersion is
`configs/uncertainty_control_dispersion_v2.csv` (one-year-ahead Poisson deviance of the
precision-weighted smoother, folds predicting 2022 to 2024). 2025.3 publishes no
interval and no reliability tier. The per-cell p10, p90 and `reliability_tier` fields
are not in the download, the lookup shards or the map. Held-out coverage: interval
calibration table in `docs/EVALUATION.md`.

## Stability

Share of cells whose crime-exposure index moves by more than 5% when the
stage-10b prior is refit on 99% of training jurisdictions (39 dropped, seed 1) and the
residual model is refit after dropping 1% of its training (jurisdiction, block group)
pairs (seed 1); tract for murder and rape, block group otherwise. Both columns are measured
on internal builds of the 2025.2 cycle, not on the published 2025.3 cells; the bagged column
on an internal `rc4` build with the 550 resident-equivalent floor everywhere. The prior and
residual model settings are unchanged in 2025.3.

| Offense | Single-fit prior and residual (2025.1 design, S2 protocol; internal build) | Bagged prior arm B, bagged residual arm 1 (internal `rc4` build) |
|---|---:|---:|
| Murder | 0.273 | 0.025 |
| Rape | 0.118 | 0.002 |
| Robbery | 0.692 | 0.268 |
| Aggravated assault | 0.643 | 0.227 |
| Burglary | 0.602 | 0.025 |
| Larceny | 0.457 | 0.122 |
| Motor vehicle theft | 0.670 | 0.231 |

The single-fit column perturbs the prior only, with the residual refit on it; the bagged
column perturbs both.

## Known limitations

- Only crime reported to and recorded by police is represented.
- Offense classification and reporting completeness vary by agency and by state.
- Neighborhood values are modeled shares of agency totals, not counted incidents.
- Allocation outside the 15 feed cities is modeled: 92.9% of block groups on larceny, 92.9% to 94.1% on the volume offenses, 98.6% of tracts on murder and 99.9% on rape (`source_mode_<offense> = modeled_transfer`).
- 233,748 of the edition's 238,193 block groups, 98.1%, sit outside the jurisdiction-panel training hull on at least one governed covariate (`extrapolation_flag` in `state/modeling/bg_mixture_experts_2025_census_residential.parquet`).
- Held-out evaluation is possible only in feed and validation cities. Feed-city performance is not a bound on performance elsewhere.
- Held-out TVD rows where a baseline arm is lower than `ours`: 6, all temporal folds against past counts (larceny at 2021; robbery, aggravated assault, burglary, larceny, motor vehicle theft at 2023), every interval spanning zero; no spatial row. AGS 2022A has lower tract TVD on murder (frozen outside set +0.042 [+0.025, +0.061]; held-out cities +0.022, interval spanning zero). Listed in `docs/EVALUATION.md`.
- The paired held-out tables leave out Atlanta and Houston and the Philadelphia offenses re-admitted in 2025.3; their unpaired results are in `docs/EVALUATION.md`.
- Agencies that filed no 2025 row receive an imputed total. Residents under a control imputed from a pooled peer unit (`level_repair_mode = pooled_silent_unit`) are 1.51% of the published population on larceny and 1.47% to 1.77% across the seven offenses; including `decayed_own_history_or_pooled` the range is 3.30% to 4.80%.
- Restricted agencies (27, `configs/reporting_regime_restrictions.csv`) carry levels from earlier admitted years or source-supported replacements, not their filed counts for the restricted years.
- Cells flagged by the current-year check (count below half its forecast and half the peer ratio, negative-binomial tail below 1/1,000), with no independent 2025 count and no confirmation; published at the precision-weighted level:

  | Jurisdiction | Offense | 2025 count | Forecast | Published level |
  |---|---|---:|---:|---:|
  | Shaker Heights, OH | Rape | 0 | 10.7 | 5.6 |
  | Union Township, NJ | Aggravated assault | 6 | 32.4 | 13.4 |
  | Wallkill, NY | Aggravated assault | 5 | 24.8 | 11.9 |
  | Auburn, NY | Motor vehicle theft | 2 | 18.6 | 7.6 |
  | Bloomfield, NJ | Rape | 0 | 8.4 | 5.2 |
  | Meridian, MS | Larceny | 180 | 532.0 | 176.0 |
  | Newnan, GA | Aggravated assault | 114 | 290.5 | 115.5 |
  | Palm Desert, CA | Aggravated assault | 34 | 104.8 | 43.5 |
  | Lincoln, CA | Burglary | 4 | 31.9 | 11.1 |
  | Winter Haven, FL | Rape | 1 | 25.4 | 8.4 |

  Source: `current_year_verification.unverified` in `state/candidates/rc3-2025.3/validation_summary.json`.
- No denominator counts visitors. Places whose offenses are driven by visitors (casinos, stadiums, tourist districts, event venues) have levels set against a resident, daytime and jobs base that leaves the visitors out. Example: Poarch Creek Tribal PD (`ALDI00300`), 430.7 larcenies against 1,671 larceny exposure units on its territory, an index of about 2,250 territory-wide.
- Campus, transit and port footprints built from facility polygons are weighted by facility land area (9 footprints), so large low-activity parcels take part of their mass.
- The Mobile, AL police jurisdiction has no polygon of its own and is not a service relationship.
- Murder and rape have no block-group rate or index.
- Sparse small-area counts carry wide internal intervals, which 2025.3 does not publish.
- LandScan USA is 2021; no later USA vintage is public.
- 2020 boundaries do not reflect later boundary changes; TIGER 2025 places and county subdivisions are not used.
- Road-length covariates for roads on block-group boundaries depend on the line-polygon intersection: 54% of block groups move by more than 5% in `road_total_length_km` between TIGER 2020 and 2025 roads.
- Alaska, Hawaii and the territories are excluded.
- An index is not an individual person's probability of victimization.
- Los Angeles has no admitted direct-incident allocation feed in this edition.

## Reproduction

- Build steps and commands: `docs/BUILD.md`
- Published columns: `docs/FIELDS.md`
- Evaluation protocol and results: `docs/EVALUATION.md`
- Level lane: `src/crimerisk/level_lane.py`, `src/crimerisk/reporting_history.py`, `src/crimerisk/controls.py`, `src/crimerisk/jurisdiction_targets.py`, `src/crimerisk/smoothed_controls.py`
- Service relationships: `src/crimerisk/service_relationships.py`, `configs/service_relationships.csv`, `configs/service_territory_exceptions.csv`
- State targets: `configs/state_anchor_exceptions.csv`, `state/controls/state_anchor_contributions_2025.csv`
- Restrictions and confirmations: `configs/reporting_regime_restrictions.csv`, `configs/current_year_confirmations.csv`
- Allocation: `src/crimerisk/allocation.py`, `src/crimerisk/mixture_allocation.py`
- Model-surface prior and residual model: `src/crimerisk/model_surface.py`, `src/crimerisk/city_residuals.py`
- Denominators and composites: `src/crimerisk/denominators.py`, `src/crimerisk/composites.py`
- Footprints: `configs/overlap_custom_footprints.csv`, `configs/pa_police_service_coverage.csv`, `scripts/build/01_tribal_territory_footprints.py`, `src/crimerisk/territory_shares.py`
- Feed admissions: `configs/city_incident_priority.csv`, `configs/city_incident_sources.csv`, `configs/city_feeds/<city>.json`
- Mixture weights and blend: `configs/mixture_ship_weights_v3.csv`, `configs/finished_share_blend_v1.csv`
- Provenance: `src/crimerisk/confidence.py`
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
