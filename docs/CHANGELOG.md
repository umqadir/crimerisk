# Changelog

## 2025.3

Data year 2025. Candidate `rc3-2025.3`; evaluation run `gold_e3_rc3b`. Asset base
`https://tiles.qqlab.io/2025.3/`.

Inputs are at their latest available vintage as checked on 2026-10-06; added since: NY DCJS
2025 (Tableau table, 2026-10-11), BIA AIAN National LAR (2026-10-11), TIGER 2020
AIANNH/AITSN, Atlanta and Houston feeds (2026-10-10 to 2026-10-11). Documented exceptions
(National Transit Map history, restricted agencies, the Houston cache, Mesa, TIGER 2020 places):
`docs/METHODOLOGY.md`, Data sources. Measured accuracy of `rc3-2025.3`: `docs/EVALUATION.md`.

| Change | Specification | Result |
|---|---|---|
| Adjudicated reporting history | Every history year runs the target-year admission sequence; a refused year is missing, not zero; a replacement is admitted at its count; one history for level repair, risk signals, smoothing, mixture training and the level holdout (`reporting_history.py`) | 2018 to 2024: 97,401 agency-offense-years rejected, 96,501 admitted partial, 91 admitted at a replacement count |
| Reporting-regime restrictions | `configs/reporting_regime_restrictions.csv`, effective from a first year until lifted; replaces build-year-keyed hard-evidence rows | 31 restrictions, 27 agencies (Chicago aggravated assault and murder from 2021, Tucson from 2024, Jackson MS from 2021, Indianapolis 2025, Columbus GA 2025, Volusia SO and Deltona 2024 to 2025, and others) |
| Clean jurisdiction-year | Complete admitted twelve-month filings carry at least 0.95 of the expected mass, expected mass past-only; unknown agencies make the year not clean (was: mean months at least 12) | — |
| Partial-year screen | Masked-gap signature also reads the Return A master's per-offense monthly profile; FDLE month counts only months with a nonzero 37-offense total | Columbus GA motor vehicle theft 2025 annualised from 5 months |
| Semantic admission | FDLE and MS TOPS sex-offense groups not admitted as rape; NY DCJS county State Police rows matched to the FBI "SP: county" ORI; NY DCJS 2025 lane | NY rape accounting 8,545 to 5,878 (rc4 state); FL and MS rape targets are CDE |
| Service relationships | Effective-dated table: `configs/service_relationships.csv` (34 rows), consolidated-footprint registry, FBI `covered_by_ori`; replaces the dead-reporting predicate; blocking territory check on the resolved footprint | Territory check 3,496 within band, 16 recorded exceptions, 0 unresolved; 12,104 municipal jurisdictions with block groups |
| Tribal territory footprints | TIGER AIANNH/AITSN plus BIA LAR land and facility blocks; every block group touching the territory; continuous prior/exposure blend s (prior x s) + (1 - s)(r x exposure) | 3,635 footprint rows, 158 agencies; Escambia AL 010539704004 larceny index 8,203 (2025.2.1) to 827 |
| County remainder partition | Ownership first, evidence = risk signal plus imputed silent mass, Dirichlet prior k = 0.5 over owned exposure; persisted as `county_remainder_partition_2025.parquet` | — |
| Precision-weighted smoothing | level = w c + (1 - w) H, w = V / (V + H); H the past-only forecast at the 2025 state level shrunk to leave-one-out past-only peers; murder against its information posterior (replaces the E1 kernel level) | One-year-ahead WAPE, origins 2023 and 2024, 102,735 cases: precision weighting 0.1651, current count 0.1678, 2025.2.1 smoother 0.1835 |
| Current-year check | 25k+ cells far below forecast and peers are flagged, not overridden; `configs/current_year_confirmations.csv` | 10 flagged cells (METHODOLOGY, Known limitations) |
| State target | CDE 2025 state estimate plus documented corrections for agencies whose own twelve-month report the FBI used; published FBI imputations held fixed; Florida motor vehicle theft 2025 a recorded exception (was: accounting total where it differs from CDE by more than 20%) | 343 cells: 294 CDE, 48 CDE plus corrections, 1 recorded exception (2025.2.1: 337 CDE, 6 accounting); Louisville murder 2025 112 (KY target 218) |
| National reference | One per offense and measure over the full surface before suppression, shared by block group, tract and composites | One-block-group tracts publish their block group's index |
| Resident floor | Per-resident index and composites from 300 residents where `special_use_type` is `ordinary`, 550 otherwise (was 550 everywhere) | Block groups 221,080 to 233,612; tracts 82,743 to 82,797; no other published cell moves |
| City feeds | Atlanta and Houston share-admitted for murder, robbery, aggravated assault, burglary, larceny and motor vehicle theft (rape truth-only); Philadelphia robbery, larceny, motor vehicle theft and murder re-admitted with excluded years; Atlanta and Houston contracts count one row per incident and offense | Feed cities 13 to 15; block groups not `modeled_transfer` on larceny 5.9% to 7.1% |
| Provenance | `source_mode` from the city posterior's own mixture ((1 - prior_fraction) x direct share / posterior share); `mixed` below a 0.6 dominant share; `source_mode_dominant_share_<offense>` read by the card | — |
| Card wording | Jurisdiction line: "modeled from a complete 2025 police filing", "modeled; the 2025 police filing was incomplete or missing", "modeled from the 2025 police filing, complete for some offenses only"; help text describes the precision-weighted level and the state total; resident floor text "fewer than 300 residents (550 in a special-use area)" | — |
| Evaluation harness | Level holdout with fold-local anchor inputs and an equivalence check; keyed truth with baseline re-scoring; validator binds level and gold runs to the edition's dependency set | `level_v12`; equivalence: all 8 stages identical |
| Evidence bundle | Build-machine paths in text members rewritten repo-relative | — |

Paired change against the published 2025.2.1 surface (`baseline_ours`, re-scored on this
run's truth), TVD, 2025.3 minus 2025.2.1, estimate [95%] (cities). The difference is the
combined effect of every change above, inputs included; Atlanta, Houston and the
re-admitted Philadelphia offenses have no 2025.2.1 cell and are not in it.

| Offense | Held-out cities | Held-out years after 2021 | Held-out years after 2023 | Frozen outside set |
|---|---|---|---|---|
| Murder | -0.0095 [-0.0176, -0.0003] (7) | +0.0011 [+0.0000, +0.0028] (7) | +0.0000 [-0.0000, +0.0001] (7) | +0.0083 [-0.0028, +0.0260] (16) |
| Rape | -0.0025 (1) | -0.0129 (1) | -0.0014 (1) | +0.0109 [-0.0006, +0.0292] (10) |
| Robbery | -0.0036 [-0.0057, -0.0010] (11) | -0.0015 [-0.0055, +0.0016] (11) | -0.0011 [-0.0039, +0.0009] (11) | +0.0031 [-0.0006, +0.0070] (19) |
| Aggravated assault | -0.0022 [-0.0034, -0.0011] (9) | -0.0111 [-0.0336, +0.0009] (9) | -0.0119 [-0.0346, +0.0002] (9) | +0.0014 [-0.0018, +0.0065] (18) |
| Burglary | -0.0033 [-0.0057, -0.0007] (11) | +0.0015 [+0.0000, +0.0038] (11) | +0.0007 [-0.0005, +0.0024] (11) | -0.0006 [-0.0053, +0.0052] (19) |
| Larceny | +0.0010 [+0.0003, +0.0016] (12) | -0.0006 [-0.0022, +0.0004] (12) | -0.0002 [-0.0009, +0.0005] (12) | +0.0012 [-0.0008, +0.0040] (19) |
| Motor vehicle theft | -0.0037 [-0.0079, +0.0015] (11) | -0.0022 [-0.0051, +0.0003] (11) | -0.0018 [-0.0042, +0.0001] (11) | +0.0018 [-0.0031, +0.0076] (19) |

`rc3-2025.3` national block-group expected counts against 2025.2.1: murder 14,064 to
14,211 (+1.0%), rape 128,295 to 121,942 (−5.0%), robbery 167,641 to 168,304 (+0.4%),
aggravated assault 806,244 to 819,286 (+1.6%), burglary 655,575 to 657,681 (+0.3%), larceny
3,875,309 to 3,885,242 (+0.3%), motor vehicle theft 677,888 to 680,141 (+0.3%).

Not adopted

| Candidate | Status |
|---|---|
| Geometric mixture, exposure expert dropped, residual on the prior (Atlanta robbery) | Not adopted |
| Smoothing each jurisdiction's share of the state total alone | Not adopted; used as the forecast part of precision weighting |
| Local-level and local-linear-trend state space (Poisson extended Kalman filter) | Diverges on small counts |
| Murder forecast at the state level | Not adopted |

Agent workflow: 615 ledger rows for edition 2025.3 in `configs/ledger/decisions.csv`.

`rc3-2025.3` surfaces: block group sha256 `0c74db61...`, tract `5c0e6be1...`.

## 2025.2.1

Data year 2025. Candidate `rc4-2025.2.1`; evaluation run `gold_rc4b`. Asset base
`https://tiles.qqlab.io/2025.2.1/`.

Inputs are at their latest available vintage as of 2026-10-06. Measured accuracy of `rc4-2025.2.1` (held-out cities, held-out years, the frozen E4 panel and AGS 2022A): `docs/EVALUATION.md`, run `gold_rc4b`.

| Change | Specification | Result |
|---|---|---|
| National Transit Map stops | Structural-cache transit columns from the NTAD layer of 2026-04-21, the latest published (2025.2: snapshot built no later than 2026-08-15) | Transit inputs at the latest vintage |
| Person-offense exposure floor | Murder, rape, robbery, aggravated assault: 300 resident-equivalents where `special_use_type` is `ordinary`, 550 otherwise (was 550 everywhere); murder and rape also 550 residents. Burglary, larceny and motor vehicle theft unchanged: 10 premises, 50 person units, 50 vehicles; 500 units where residents are below 50 | Block groups with a published crime-exposure index: robbery 221,387 to 235,020 (92.9% to 98.7%), aggravated assault 221,606 to 234,673 (93.0% to 98.5%), Overall 218,764 to 234,003, Violent 218,938 to 234,185. Tracts: murder 82,737 to 82,742, rape 82,735 to 82,743, robbery 83,025 to 83,076, aggravated assault 82,994 to 83,039, Overall 82,723 to 82,741, Violent 82,724 to 82,742. Block-group index p99.9 / maximum: robbery 1,924 / 6,825 to 2,074 / 6,762; aggravated assault 1,163 / 4,612 to 1,322 / 4,533 |
| Map card, withheld index | Shows the area's expected annual count, named by area type; stated on the methods page | — |

`rc4-2025.2.1` surfaces: block group sha256 `ef2f504f...`, tract `5631ac45...`.

## 2025.2

Data year 2025. Candidate `rc4-2025.2`; evaluation run `gold_rc4a1`. Asset base
`https://tiles.qqlab.io/2025.2/`.

Evaluation run `gold_rc4a1`; superseded by the 2025.2.1 run.

Data refreshes (inputs at the 2026-10-06 vintage)

- FBI CDE `estimated_crimes_1979_2025`: 2025 rows unchanged; 2024 rows were an earlier vintage, 455 of 520 cells changed (national motor vehicle theft 2024: 880,327 to 870,686). CIUS 2020 to 2025 bundles, `reta-2025.zip`, `nibrs-2025.zip` and the NIBRS 2024 and 2025 tables byte-identical to CDE (Last-Modified 2026-08-14).
- FBI NIBRS agency tables 2025 added: 90,083 rows (territories dropped); parser reproduces 2024 exactly.
- CDE agency rosters re-pulled: 2025 +20 / −5 ORIs, 20 NIBRS-status changes; 2024-labelled copy +116 / −65.
- FDLE FIBRS offense detail 08-2026: 2024 restated (2,517 counts in 112 agencies), 2025 added (254 agencies). MS TOPS: 2024 221 counts restated, 2025 added (845 rows, 156 agencies). CT DESPP re-export: 2025 +4 offenses in 4 rows.
- City feeds re-pulled, all release cities except Mesa (endpoint gone; frozen cache 2020 to 2025). Pools: St. Louis 2021 to 2025, Baltimore 2018 to 2024 plus NIBRS 2025, Denver 2021 to 2025, Minneapolis 2019 to 2025, Washington volume offenses 2018 to 2025 (was 2025 only).
- Philadelphia 2022 was truncated in the previous cache: aggravated assault 5,480 to 8,275, burglary 3,851 to 5,862.
- New York feed: the 2026-03-22 historic download was a truncated export (2018 to 2024 had 11 to 4,410 located incidents a year). Now pulled by report year with count verification; New York murder 2018 to 2019 withheld (precinct masking). Validation surface New York located incidents 229,313 to 1,684,563.
- Overture places 2026-06-17.0 to 2026-09-23.1: CONUS and DC destination points 2,074,449 to 1,844,679 (−11.1%); 123,730 of 238,193 block groups change destination count.
- Structural cache: TIGER roads 2020 to 2025, NLCD 2023 to 2025. National Transit Map stops: snapshot built no later than 2026-08-15 (moved to the 2026-04-21 layer in 2025.2.1).
- Not refreshed: LandScan USA 2021 (no later USA vintage public); TIGER places and county subdivisions stay 2020 (165 of 12,798 municipal geoids recoded or dissolved in TIGER 2025, 149 of them Connecticut towns); ACS 2020 to 2024 (2021 to 2025 not released); LODES 2023.

Adopted model and denominator changes

| Change | Specification | Held-out result |
|---|---|---|
| S2: bagged stage-10b prior, arm B | Mean of 10 fits on keyed 80% jurisdiction subsamples; depth 3, min leaf 100, l2 10 (was 5, 20, 1) | `gold_rc3` vs `gold_rc2`, spatial Δ TVD: aggravated assault −0.0064 [−0.0094, −0.0034], burglary −0.0018 [−0.0043, +0.0006], larceny −0.0027 [−0.0058, +0.0003], motor vehicle theft +0.0027 [+0.0002, +0.0047], murder +0.0009 [−0.0013, +0.0034] (Δ log score −0.0004 [−0.0060, +0.0043]), robbery −0.0015 [−0.0047, +0.0020] |
| S3: bagged residual model, arm 1 | Mean of 10 fits on keyed 80% city subsets; depth 5, min leaf 20, l2 1 (unchanged) | In `rc4-2025.2`; selected over arm 2 on the E4 model fold (Not adopted, below) |
| M1: larceny finished-share blend | 0.8 model, 0.1 population, 0.1 primary exposure on model-only jurisdictions; other offenses unchanged | Larceny Δ TVD −0.0041 [−0.0065, −0.0016] (vs `gold_b0`) |
| M2a: murder violent-incident tract prior | Feed-city murder prior from the city's aggravated-assault and robbery tract share, rolling-origin strength | Murder Δ log score +0.2226 [+0.0124, +0.5509] (temporal 2021), +0.1527 [+0.0007, +0.4049] (temporal 2023) |
| D1: Census residential leg | Person-exposure residential leg is release Census population (was LandScan night) | Non-inferior: spatial Δ TVD aggravated assault +0.0008 [+0.0003, +0.0014], robbery +0.0004 [+0.0001, +0.0008], larceny −0.0000 [−0.0003, +0.0003]; block groups of at least 500 residents with primary denominator below 0.2x residents: robbery 347 to 0, murder 472 to 0, larceny 114 to 0 |
| D2: vehicle floor | Motor vehicle theft denominator at least 0.2511 vehicles per person present | Vehicle denominators raised in 4,830 tracts and 7,394 block groups; national 347,892,586 to 350,597,394 |
| D3: per-resident floor | Per-resident index and resident composites published at 550 residents or more (was 50) | Per-resident publication: 82,743 tracts, 221,080 block groups |
| State rake | Smoothed levels raked per state and offense to the CDE 2025 state estimate; accounting total where the two differ by more than 20%; national total = sum of state targets | In `rc4-2025.2` |
| Burglary group-quarters term | Fifth NNLS premises term, 2020 DHC group-quarters residents: 0.266 premises per resident | In `rc4-2025.2` |
| Exposure floor | Murder, rape, robbery and aggravated assault crime-exposure indexes at 550 resident-equivalents or more; murder and rape also 550 residents. Replaces the 50-resident / 500-unit floor for those offenses. Burglary, larceny and motor vehicle theft keep the rc3 floors: 10 premises, 50 person units, 50 vehicles; 500 units where residents are below 50 | In `rc4-2025.2` |
| B2: state-police footprint share | A state-police unit keeps on its footprint only the share of its FBI population served that the footprint's residents cover | 2,546 offenses moved (`rc4-2025.2`); national and state totals identical |

State rake: 343 state-offense cells; 6 accounting targets (Florida motor vehicle theft,
Florida rape, Kentucky murder, Louisiana rape, Mississippi rape, New York rape). National
totals: murder 14,091.40, rape 128,758.63, robbery 167,768, aggravated assault 807,711,
burglary 657,029, larceny 3,884,438, motor vehicle theft 678,780.52.

Burglary premises coefficients (`rc4-2025.2`): 24.346 per destination POI, 0.323 per
retail job, 0.044 per industrial job, 0.266 per group-quarters resident; national total
181,632,096 premises.

Not adopted

| Candidate | Result |
|---|---|
| M4: rolling-origin feed posterior strength | No offense wins both temporal folds; temporal 2023 motor vehicle theft −0.0102 [−0.0239, −0.0010] |
| M3: adaptive hierarchical pooling for silent units | Dropped-subset deviance lower on 1 of 7 offenses; median abs log ratio 0.3900 to 0.3901 |
| M6: drop `proxy_review` socioeconomic features | Withdrawn before any run |
| M7: admit non-racial protected-class features | Specified; not run in 2025.2 |
| S3 arm 2: bagged residual, depth 3 / min leaf 100 / l2 10 | E4 model fold Δ TVD against `gold_rc3` on the same frozen truth: robbery +0.0095 [+0.0066, +0.0123], aggravated assault +0.0073 [+0.0037, +0.0107], larceny +0.0035, motor vehicle theft +0.0025; arm 1: +0.0009, −0.0016, −0.0008, −0.0052. Worse than arm 1 on all four |
| D3 precision test (Poisson RSE < 0.30) | Withdrawn; the 550-resident minimum is kept |
| Target-year footprint relationships from 2024 batch headers | Broke Pennsylvania PCCD service links; Jacksonville fill 26,766 to 50,524 |
| Mesa replacement feed (`t445-fism`) | Onboarded on a branch; not in 2025.2 |
| TIGER 2025 places and county subdivisions | Not migrated |

Correctness

- New York timestamps parse identically whether blank fields arrive as NaN or None (4,184 complaints had different midpoints; 2 changed year).
- A state-publication agency-year below 0.25x a full-year FBI report for the same agency-year is not a candidate lane (FDLE zero-fills unreported months; Seminole SO 2024: 0 against FBI 1,810).
- Miami-Dade Sheriff's Office matches the former Miami-Dade PD (`FL0130000`).
- Block-group rook-edge producer committed (`scripts/build/build_bg_rook_edges.py`, 655,942 edges).
- Stage 7 stops when an enabled feed city produces no rows, and when an admitted city-offense year falls below 1/3 of its series median. Socrata downloads retry on `IncompleteRead` and connection reset.
- Overture pull selects `categories.primary` only when the release schema has it; feature-identical on 2026-08-19.0.
- National anchor: the smoothed controls reconciled to a kernel-weighted history of CDE national estimates; 2025.1 surface plus unlocated mass over CDE 2025 was murder 1.166, rape 1.060, robbery 1.226, aggravated assault 1.062, burglary 1.134, larceny 1.071, motor vehicle theft 1.192. Replaced by the data-year anchor, then by the state rake.
- Structural-zero lock requires the target year's own filed zero vector on at least 11 months: in 2025, 159 of 404 previously locked agencies had filed 1,372 offenses.
- Connecticut resident-trooper towns: DESPP town lane carries 2021 to 2025; CTCSP0000 residualized on town reports or fills (2025: FBI 1,300.8, towns 1,010, residual 316.9).
- Partial-year floor compares partial offenses with their own history and refuses only the partial rows.
- Zero-total municipal towns take their covering pool's rate: 185 towns, 1,126.2 expected offenses.
- Covered agencies carry no 2025 numerator: 24 ORIs, 4,354 offenses removed as second numerators.

Evaluation fixes

- F1: gold folds build with the fold's own larceny blend and stage-10b prior; run manifests carry a dependency fingerprint and resume is refused on any difference; the release validator binds the published surfaces to the gold run whose model fold recorded their hashes; the remainder holdout uses production's eligibility, pool and ratio (2025: 56 cases, was 144; median abs log ratio 0.245, was 0.535); the combined level runner reads `mask_share`.
- F2: stage 11b is the one residual-model lifecycle; freshness stamps fail closed; the stage-10b stamp covers every model-surface setting and the feature-catalog digest; the input manifest substitutes the target year.
- The E4 model fold is labelled a tuning corpus for `ours`.

Stability

Share of cells whose crime-exposure index moves by more than 5% under a 99% prior refit
and a 1% residual-training drop (S3 protocol (C), measured on an internal `rc4` build with the 550 resident-equivalent floor everywhere, not on the published cells):

| Offense | S3 arm 1 (adopted) | S3 arm 2 |
|---|---:|---:|
| Murder | 0.025 | 0.025 |
| Rape | 0.002 | 0.002 |
| Robbery | 0.268 | 0.134 |
| Aggravated assault | 0.227 | 0.067 |
| Burglary | 0.025 | 0.025 |
| Larceny | 0.122 | 0.021 |
| Motor vehicle theft | 0.231 | 0.097 |
| Mean | 0.129 | 0.053 |

S2 (prior perturbation, residual refit), mean share: single fit 0.494, arm A 0.334, arm
B 0.266.

Site and workflow

- Per-index help text on the card and methods page; crime exposure is the default measure and the card shows both values.
- Per-measure publication status, mobile bottom sheet, first-paint fix.
- Exception ledger (05b) and evidence bundle (05c).
- Judgment steps in onboarding and adjudication are drafted and reviewed by agents and recorded in `configs/ledger/decisions.csv`; the build reads only ledger-approved rows.


## 2025.1.1

Data year 2025. Data, tiles and download files are 2025.1's, served from `2025.1/`.

Level-lane holdout (`scripts/eval/level_holdout.py`)

- The pooled silent-unit fallback is built from each fold's training panel, with the fold's masked agencies excluded. It was built from the unmasked panel.
- `tests/test_level_holdout_leakage.py` fixture, unmasked panel: raising the held-out agency's hidden 2024 count from 1,000 to 2,000 raises its fallback prediction from 550 to 1,050. Training panel: 100 in both cases.
- The pool panel carries no population column, so every pooled rate is undefined and filled with 0. All 459 ladder-dropped rows are predicted 0 in every configuration of `level_v1` and of the rerun `level_v2`. The leak therefore moved no `level_v1` number. The zero-prediction defect is not fixed in 2025.1.1.

Release evaluation check (`_check_release_evaluation`)

- Replaces the requirement that the measurement harness recommend `allocator_expansion_first`.

| Requirement | 2025.1.1 status | Result |
|---|---|---|
| `gold_v53r2` results and manifest present; run id and build year match | Blocking | Pass |
| Evaluated candidate equals the published candidate `v53-2025` | Blocking | Pass |
| Spatial TVD: `ours` below population for every offense except rape | Reported, not blocking | Fail: motor vehicle theft, 0.298 against 0.287 |

Evaluation document

- Temporal folds: feed-truncation experiments; every other input is the 2025 edition's.
- Spatial folds: four cities per offense, one for rape. Transfer outside the feed cities is listed as not measured.
- The baseline table adds the past-counts city count and `ours` TVD on the past-counts cities. The 2025.1 table compared past counts on fewer cities with `ours` on all cities.
- Rows where a baseline has lower TVD than `ours`: spatial motor vehicle theft, population 0.287 against 0.298; spatial burglary, primary exposure 0.280 against 0.287; on the past-counts cities, 2021 aggravated assault 0.356 against 0.366, 2021 motor vehicle theft 0.311 against 0.316, 2023 aggravated assault 0.377 against 0.399, 2023 motor vehicle theft 0.347 against 0.353, 2023 robbery 0.406 against 0.415.

Public copy

- Methods page: the crime-exposure denominator is described as offense-specific mixtures of modeled population and activity proxies, rescaled to national resident population. The "40 residents and 30,000 daily visitors" example is removed.
- Methods page: the claim that a high special-use score "usually" reflects few residents is removed.
- Methods page: composite support by measure; Technical methods link to the documents in `umqadir/crimerisk`.
- METHODOLOGY: composite support table per measure and support; public composite names mapped to internal fields.
- BUILD names `gold_v53r2` as the release run. Stage 10 is the normalizer surface and stage 11 the expert table in BUILD, EVALUATION and the 2025.1 entry below. Population estimates vintage is 2025 (`co-est2025-alldata.csv`).
- README and the download row link the download page; `https://tiles.qqlab.io/2025.1/downloads/` has no listing.

Card

- Denominator line: "Denominator: residents" on Per resident; people present, premises, vehicles present, or people and destinations present, each marked modeled, on Crime exposure; each offense's own modeled base on composites.
- Jurisdiction-total line from a new shard array, per offense: 1 where `level_admission_status = valid_complete_year` and `level_repair_mode = none`, 0 otherwise, null where the status is missing. Text: "reported for 2025", "estimated, not reported in full for 2025", or reported on some offenses and estimated on others. No line where every value is null. Block-group offense values: 1,033,359 reported, 117,809 estimated, 516,183 null.
- Estimated offense count shown as a whole number; "under 1" below 0.5.

Search

- Three messages: lookup service error or non-200 response; no result; results only outside the 48-state-and-DC box (-124.85, 24.35, -66.85, 49.45). Results outside the box are not listed.

Build

- `frontend/build/crschema.py` and `frontend/build/03_tiles.sh` default the tiles root to `crimerisk-tiles` beside the repository, not a home-directory path.

## 2025.1

Data year 2025. Previous release: 2026-08-18.

- The level lane consumes the FBI 2025 Return A master file.
- Public measures reduced from nine to two: Crime exposure and Per resident.
- Density, harm-weighted burden and multi-offense relative scores left the public product.
- Composites are Overall, Violent and Property, as national expected-count-share-weighted averages of the offense indexes.
- A composite is suppressed when any component it uses is suppressed; support by measure is in METHODOLOGY.
- The suppressed rare-offense composite term no longer takes a parent-tract fill.
- Murder and rape route to tract at every zoom; rare-offense density is no longer published.
- The legend is seven verbal classes, replacing 15 fixed log-spaced breaks.
- The public download schema is 64 columns at block group and 63 at tract, from crschema.py.
- Index intervals and the reliability tier are not published in 2025.1: held-out coverage of the internal intervals is below nominal.
- No nowcast edition is published.
- Custom footprint overlap mass is raked per ORI to that agency's ledger control.
- Unplaceable footprint mass routes to the unlocated-mass table; a deviation above 5% blocks release.
- County rollups are count-weighted rates with a floor of 2 tracts and 2,500 residents.
- Jurisdiction and state comparison values are count-weighted with a 2 block-group floor.
- Tiles are three archives, counties z3 to z7, tracts z5 to z12, block groups z8 to z12, with tile-size limits enforced.
- Tile features carry 25 attributes; card fields moved to GEOID lookup shards.
- The map gained submit-only search, two-location compare, a mobile sheet and a docked desktop panel.
- `scripts/eval/gold_eval.py` adds leave-one-feed-city-out and feed-year-truncation folds.
- Docs are README, METHODOLOGY, BUILD, FIELDS, EVALUATION, CHANGELOG; PIPELINE.md, FIELD_DICTIONARY.md and SPEC.md were removed.
- Nothing in the build is pinned. The stage-10 normalizer surface and stage-11 expert table, which 2025.1 previously shipped as reviewed 2026-08-18 artifacts, are retrained from current inputs; the release is reproducible from the documented stage order.
- The documented stage order now builds the exposure normalizers before the mixture experts. The mixture's exposure expert reads the normalizer surface, so the old order failed on a fresh build and left a stale expert table on an incremental one.
- Retraining the mixture experts against the refreshed crosswalk and jurisdiction estimates moved the feature panel from 10,863 to 10,881 jurisdictions and the training rows from 8,282 to 8,286. National anchors, published cell counts and the published universe are unchanged; block-group indexes move by under 20 index points at p99 on the composites.
- The exposure normalizer surface retrains byte-identically, and its summary now records the residential leg it was built with.
- The person-exposure residential leg is LandScan nighttime population. The Census-residential arm (`--residential-leg-source census_release_population`) stays an opt-in sensitivity arm under its own artifact and normalizer version and is not published.
