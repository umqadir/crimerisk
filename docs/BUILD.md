# Build

Edition 2025.3. Every command runs from the repository root.

```bash
uv sync
```

## Inputs

Raw inputs are not in this repository. They are staged under `data/` and `state/`
before the first stage runs. Vintages as checked on 2026-10-06; documented exceptions in
`docs/METHODOLOGY.md` (Data sources).

| Input | Source | Vintage | Staged at |
|---|---|---|---|
| FBI UCR Return A master file, 2025 | FBI UCR | reta-2025.zip, CDE Last-Modified 2026-08-14 | `data/FBI-UCR-Return-A-Parsed-2025-20260921/return_a_master_annual_2025.parquet` |
| FBI CIUS annual tables | FBI | 2020 to 2025 bundles, Last-Modified 2026-08-14; 2018 to 2019 legacy tables | `data/FBI-CIUS-Annual/<year>/raw/` |
| FBI CDE state and national estimates | FBI | 1979 to 2025, Last-Modified 2026-08-14 | `data/FBI-CDE-Estimates-1979-2025/estimated_crimes_1979_2025.csv` |
| FBI NIBRS agency tables, 2024 and 2025 | FBI | Last-Modified 2026-08-14 | `data/FBI-NIBRS-Tables-<year>/parsed/nibrs_offense_type_by_agency_<year>.parquet` (`scripts/pull/parse_fbi_nibrs_agency_tables.py --year <year>`) |
| FBI CDE agency rosters, 2024 and 2025 | FBI CDE API | Pulled 2026-10-06 | `data/FBI-CDE-Agency-Rosters-<year>/` |
| NIBRS segment files, Kaplan | ICPSR | V11, 1991 to 2024 | `data/NIBRS-Kaplan-1991-2024/` |
| SRS offenses known, Kaplan | ICPSR | V22, 1960 to 2024 | `data/SRS-Kaplan-1960-2024/` |
| State publication, FDLE FIBRS | FDLE | Offense detail workbook 08-2026: 2024 to 2025 | `data/FDLE-FIBRS-2024/` |
| State publication, MS TOPS | MS DPS | 2024 to 2025 | `data/MS-TOPS-2024/raw/` |
| State publication, NY DCJS | NY DCJS | 2021 to 2024 Socrata (rowsUpdatedAt 2025-11-25); 2025 Tableau export 2026-10-11 | `data/NY-DCJS-2024/`, `data/NY-DCJS-2025/` |
| State publication, CT DESPP | CT DESPP | 2021 to 2025, re-exported 2026-10-06 | `data/CT-DESPP-2024/` |
| City incident feeds, 15 cities | City open-data portals | Re-pulled 2026-10-06; Atlanta 2026-10-10; Houston: cache of 2026-10-11 (identical to the files served on recheck); Mesa: frozen cache, 2020 to 2025 | `data/city-incidents/<city>/` (`scripts/pull/pull_city_feed.py --city <key>` for contract feeds) |
| 2020 DHC group quarters by block group | Census Bureau | 2020 | `data/Census-2020-DHC/parsed/dhc_group_quarters_block_groups.parquet` |
| TIGER/Line 2020 | Census Bureau | 2020 | `data/tiger_tabblock20/`, `data/tiger_bg/`, `data/tiger_tracts/`, `data/tiger_counties/`, `data/tiger_places/`, `data/tiger_cousub/`, `data/tiger_aiannh/`, `data/tiger_aitsn/` |
| BIA AIAN National LAR | BIA | Live layer, retrieved 2026-10-11 | `data/bia_lar/bia_aian_national_lar.geojson` (`scripts/pull/pull_bia_lar.py`) |
| Population estimates | Census Bureau | Vintage 2025 | `data/Census-PopEst-2020-2025/co-est2025-alldata.csv` |
| ACS 5-year | Census Bureau | 2020 to 2024 | `data/ACS-5yr-2020-2024/parsed/` |
| LEHD LODES8 WAC | Census Bureau | 2023 | `data/LODES/parsed/` |
| LandScan USA | Oak Ridge National Laboratory | 2021 | `data/LandScan-USA/` |
| Overture places | Overture Maps | 2026-09-23.1 | `data/Overture-Places/parsed/` |
| Block-group structural feature cache | Derived: TIGER 2025 roads, HPMS 2024, National Transit Map stops (layer of 2026-04-21), NLCD 2025, NCES EDGE 2024-25, CMS hospitals | Rebuilt 2026-10-06; transit columns restored 2026-10-09 | `state/modeling/bg_structural_risk_features_2024.parquet` |
| AGS 2022A tract index (evaluation comparator only) | AGS | 2022A | `state/cache/ags_2022a/ags_2022a_tract.parquet` |

`src/crimerisk/required_inputs.py` holds the expected path and presence check for
each input; `build-input-manifest --year <year>` substitutes the target year and counts an
empty directory as missing. The structural-zero screen is a governed config,
`configs/level_lane_structural_zero_screen.csv`. Source notices: `ATTRIBUTION.md`.
Source URLs, pull commands, liveness and archive copies: `docs/INPUTS.md`.

The San Francisco feed download needs the certifi CA bundle:

```bash
SSL_CERT_FILE="$(uv run python -m certifi)" uv run python main.py build-city-incident-shares \
  --year-start 2018 --year-end 2025 --force-source-refresh
```

## Stage order

Run the stages in this order. Each command writes into `state/` and is reused by the
next stage unless a `--force-*` flag is passed.

| # | Stage | Command | Runtime, 2025.3 |
|---|---|---|---|
| 1 | Input manifest | `uv run python main.py build-input-manifest --year 2025` | 2 s |
| 2 | Agency master | `uv run python main.py build-agency-master` | 6 s |
| 3 | Reference layers | `uv run python main.py build-reference-layers` | 15 s |
| 4 | State-publication input | `uv run python main.py build-state-publication-inputs --year 2025` | 9 s |
| 5 | Observations panel | `uv run python main.py build-observations --year-start 2018 --year-end 2025` | 144 s |
| 5a | Geometry and crosswalks | `uv run python main.py build-geometry` | 26 s |
| 6 | Reporting regimes | `uv run python main.py build-reporting-regimes --year-start 2018 --year-end 2025` | 360 s |
| 7a | City feed inputs | `uv run python scripts/build/city_incident_inputs.py` | 1 s |
| 7 | City incident shares | `uv run python main.py build-city-incident-shares --year-start 2018 --year-end 2025` | 9 s |
| 7b | Validation-surface feed rows | `uv run python scripts/build/refresh_validation_surface_feed_rows.py --year 2025` | 1 s |
| 8 | Accounting controls | `uv run python main.py build-controls --year 2025` | 455 s |
| 9 | Smoothed controls and state rake | `uv run python main.py build-smoothed-controls --year 2025` | 17 s |
| 10 | Exposure normalizers | `uv run python main.py build-exposure-normalizers --year 2025` | 2 s |
| 10b | Model-surface prior | `uv run python main.py build-model-prior --year 2025` | 492 s |
| 11 | Mixture experts | `uv run python main.py build-mixture-experts --year 2025` | 26 s |
| 11b | Residual model | `uv run python main.py build-residual-model --year 2025` | 288 s |
| 12 | Outputs | `uv run python main.py build-outputs --year 2025 --candidate-run <run-id>` | 608 s |
| 12a | Block-group rook adjacency (read by the exception ledger) | `uv run python scripts/build/build_bg_rook_edges.py` | 89 s |
| 12b | Exact-point QA | `uv run python scripts/diagnostics/build_city_feed_exact_point_qa.py --year 2025` | 115 s |

Stage 12 runs allocation, composites, the uncertainty layer and the output writer in
one process. It writes to `state/candidates/<run-id>/`.

Runtimes are wall time from the `rc3-2025.3` build on an Apple silicon laptop, every stage a
fresh process from one commit, from a `state/` carrying the staged artifacts.
Stage 7 is timed without `--force-source-refresh`.

| Artifact | Stage | Written to |
|---|---|---|
| Census-residential exposure normalizers | 10 | `state/modeling/bg_exposure_normalizers_2025_census_residential.parquet` |
| Model-surface prior, feature catalog, summary | 10b | `state/modeling/bg_prior_long_2025_arm_b.parquet`, `bg_prior_long_2025_arm_b_features.parquet`, `bg_prior_long_2025_arm_b.json` |
| Mixture expert table | 11 | `state/modeling/bg_mixture_experts_2025_census_residential.parquet` |
| Residual model | 11b | `state/modeling/city_residual_model_2025.joblib`, `.deps.json`, `.manifest.json` |
| Block-group rook edges | 12a | `state/geometry/bg_rook_edges_2020.parquet` |
| Adjudicated reporting history | 8 | `state/controls/reporting_history_2025.parquet` |
| Service-territory check | 8 | `state/controls/service_territory_check_2025.csv` |
| State-target contributions and sensitivity | 9 | `state/controls/state_anchor_contributions_2025.csv`, `state_anchor_sensitivity_2025.csv` |

| Parameter | Stage 10b prior | Stage 11b residual |
|---|---|---|
| Model | HistGradientBoosting, mean of 10 bagged fits | HistGradientBoosting, mean of 10 bagged fits |
| Bag membership | sha256(seed, k, jurisdiction) < 0.8 | sha256(seed, k, jurisdiction) < 0.8, by city |
| Seed | 20260926 | 20260927 |
| Depth, min leaf, l2 | 3, 100, 10 | 5, 20, 1 |
| Iterations, learning rate | 500, 0.03 | 500, 0.03 |
| Rebuild flag | `--force-bg-prior-rebuild` | `--force-residual-model-rebuild` |

Stage 12 defaults are the release configuration. `--no-enable-mixture-allocation`,
`--no-enable-exposure-ensemble` and `--no-enable-soft-shrinkage` turn those lanes off;
`--no-enable-soft-shrinkage` on stage 12 requires the same flag on stage 11. Stage 12
stops if the stage-10b prior or the stage-11b model is stale against its stamp. A build
with `--exclude-feed-city` or `--feed-year-end` reads its own prior and residual model
(`--bg-prior-path`, `--feed-inputs-dir`) and refuses the release ones.

Stage 7 stops when an enabled feed city produces no rows, or when an admitted
city-offense year falls below 1/3 of its series median.

After a stage-7 change that moves a feed city, refresh the validation surface's feed
rows and rerun from stage 10b:

```bash
uv run python scripts/build/refresh_validation_surface_feed_rows.py --year 2025
uv run python main.py build-model-prior --year 2025
uv run python main.py build-mixture-experts --year 2025
uv run python main.py build-residual-model --year 2025
uv run python main.py build-outputs --year 2025 --candidate-run <run-id>
```

Running this order again is a no-op on the artifacts. Stages 5, 6 and 8 redo their
work and land on the same bytes; every other stage reports its artifact current and
returns in about a second. Measured on the 2025.1 assembly: a second pass over stages
1 to 11 left all 218 parquet and JSON artifacts under `state/modeling`,
`state/controls`, `state/geometry`, `state/reference` and `state/cache/mixture`
bit-identical.

A stage decides whether to reuse its artifact from the dependency stamp beside it
(`<artifact>.deps.json`), which records the content digest of every input under its
repo-relative name. A missing stamp, or one that no longer describes the artifact on
disk, means the artifact is rebuilt.

## Governed footprints and relationships

Tribal territory footprints, service relationships, territory exceptions, reporting-regime
restrictions, state-target exceptions and current-year confirmations are governed configs:
they change only through ledger-approved rows (`scripts/agents/`, `configs/ledger/decisions.csv`).
The tribal footprint proposal is built from TIGER 2020 AIANNH/AITSN, the BIA LAR layer, 2020
blocks, LODES WAC/RAC blocks, Overture points and LandScan:

```bash
uv run python scripts/pull/pull_bia_lar.py
uv run python scripts/build/01_tribal_territory_footprints.py --out-dir <proposal dir>
```

## Outputs

| File | Contents |
|---|---|
| `state/candidates/<run-id>/crimerisk_block_group_2025_ags_core.parquet` | Block-group surface |
| `state/candidates/<run-id>/crimerisk_tract_2025_ags_core.parquet` | Tract surface |
| `state/candidates/<run-id>/manifest.json` | Resolved argv, source SHA, input mtimes, build summary |
| `state/candidates/<run-id>/validation_summary.json` | Validator result |
| `state/candidates/<run-id>/unlocated_mass_2025.parquet` | Mass that could not be placed on a footprint |
| `state/candidates/<run-id>/city_posterior_diagnostics_2025.parquet` | Feed-city posterior weights per city and offense |
| `state/candidates/<run-id>/county_remainder_partition_2025.parquet` | Evidence and prior mass of each county-remainder split |
| `state/candidates/<run-id>/extremes_lineage_2025.csv`, `.md` | Lineage of the highest published indexes |

`rc3-2025.3` surfaces: block group sha256 `0c74db61...`, tract `5c0e6be1...`.

## Validator

```bash
uv run python scripts/diagnostics/build_city_feed_exact_point_qa.py --year 2025 --skip-if-current
uv run python scripts/diagnostics/validate_release_outputs.py --year 2025 \
  --state-output-dir state/candidates/<run-id> \
  --summary-out state/candidates/<run-id>/validation_summary.json
```

The validator exits non-zero on any issue. Its blocking blocks are total-lane integrity,
allocation coherence, index coherence, spatial artifacts, the feature and redlining
audit, frontend readiness, and the release evaluation's presence and candidate
binding, the level holdout's binding to the edition's inputs, and the one-block-group tract
reference. The current-year check's flagged cells are listed (`current_year_verification`).
Promotion to `state/output/` requires a validator run with no issue.

| Parameter | Value |
|---|---|
| Exact-point QA artifact | `state/qa/city_feed_exact_point_concentration_2025.csv`; rerun after stage 7 |
| Exact-point block | One coordinate at or above 0.5% of a city-offense's located incidents, unless listed in `configs/city_feed_exact_point_exceptions.csv` |
| Burglary commercial-share gradient, modeled-transfer regime | Blocks outside `[0.8, 1.45]` |
| Burglary commercial-share gradient, direct-city regime | Reported outside `[0.8, 1.3]` (`build_manifest.burglary_commercial_gradient_direct_findings`) |

`_check_release_evaluation`:

| Requirement | 2025.3 |
|---|---|
| Build manifest under `--output-dir` names a candidate run id (`run.run_id`) | Blocking |
| A completed run under `state/eval` whose scoring-only model fold recorded the sha256 of this output's block-group and tract `ags_core` surfaces, for a candidate directory of that id (latest such run is used) | Blocking |
| That run's manifest carries a dependency fingerprint (fold-scoped harness) | Blocking |
| Its `results.csv` matches the manifest's `results_sha256`; run id and build year match | Blocking |
| Spatial TVD: `ours` below the population arm for every offense except rape | Reported in `release_evaluation.failures`, not blocking (`RELEASE_GOLD_TVD_BLOCKING = False`) |

`scripts/release/promote_candidate.py` binds the snapshot to the promoted candidate by
hash. Rape is reported and not gated: one eligible fold city.

## Tiles and site

`frontend/build/snapshot_config.env` names the candidate parquets the snapshot is
frozen from. Set it before running these steps. Large artifacts are written to
`CRIMERISK_TILES_ROOT`, not into the repository.

| # | Step | Command | Runtime, 2025.3 |
|---|---|---|---|
| 1 | Public surfaces and benchmarks | `uv run python frontend/build/01_extract.py` | 4 s |
| 2 | Geometry join, GeoJSONSeq | `uv run python frontend/build/02_geometry.py` | 45 s |
| 3 | PMTiles archives | `bash frontend/build/03_tiles.sh` | 8 min 8 s |
| 4 | Lookup shards and benchmarks | `uv run python frontend/build/04_shards.py` | 18 s |
| 5 | Download package | `uv run python frontend/build/05_download.py` | 82 s |
| 5b | Cross-jurisdiction exception ledger | `uv run python frontend/build/05b_exception_ledger.py` | 1 s |
| 5c | Evidence bundle | `uv run python frontend/build/05c_evidence_bundle.py --edition 2025.3 --candidate state/candidates/rc3-2025.3 --gold state/eval/gold_e3_rc3b --level state/eval/level_v12` | 9 s |
| 6 | Stage the site | `uv run python frontend/build/06_stage_site.py` | 5 s |
| 7 | Browser verification | `uv run --with playwright python frontend/build/07_verify.py` | 51 s |

Step 7 needs Playwright and its Chromium build in the running interpreter; install the
browser once with `playwright install chromium`.

Step 6 writes the public `https://tiles.qqlab.io/2025.3/` URLs into the staged site and
fills the per-index help text from the snapshot (composite weights from national
expected counts, the fitted group-quarters premises weight). Step 7 measures the map
against a local range server, so stage the site twice: once with
`CRIMERISK_ASSET_BASE_URL=http://127.0.0.1:8791/data/tiles` for step 7, then again with
no override to produce the tree that is uploaded.

Step 3 builds three archives: counties z3 to z7, tracts z5 to z12, block groups z8 to
z12. Tile-size limits are enforced, with a 250,000-byte p95 budget per tile. Step 7
serves the staged site, drives it at 1440x900 and 390x844, and records screenshots,
first meaningful data paint and bytes transferred.

## Field documentation

```bash
uv run python docs/make_fields.py
```

Regenerates `docs/FIELDS.md` from `frontend/build/crschema.py`. Run it whenever the
public schema changes.

## Gold evaluation

```bash
uv run python scripts/eval/gold_eval.py --run-id <eval-run-id> --dry-run
uv run python scripts/eval/gold_eval.py --run-id <eval-run-id> \
  --score-candidate state/candidates/<run-id> --baseline-run <baseline-eval-run-id>
```

`--dry-run` prints the fold list and the build command for each fold without building.
Folds are selected with repeated `--folds` arguments from `spatial`, `temporal` and
`model`; all three run by default. Spatial folds rebuild one surface per feed city
with `--exclude-feed-city`; `--cities` restricts that set to named feed-city keys.
Temporal folds rebuild with `--feed-year-end 2021` and `--feed-year-end 2023`. The
model fold scores `--score-candidate` on the E4 panel (`configs/e4_panel_v1.json`).
Each held-out fold builds its own stage-10b prior, residual model and finished-share
blend. Bootstrap resampling is clustered by city, 1,000 iterations by default,
settable with `--bootstrap-iterations`. `--no-ags` drops the AGS comparator. Fold
surfaces are deleted after scoring.

Every fold is given its own `--feed-inputs-dir` under
`state/eval/<eval-run-id>/inputs/<fold-id>/`. At the end of the run the harness
re-hashes the shared city-feed artifacts it read and fails if any changed. A resume is
refused when the run manifest's dependency fingerprint differs.

Results are written to `state/eval/<eval-run-id>/`:

| File | Contents |
|---|---|
| `results.csv`, `results.md` | One row per offense by fold type, arm and metric, with intervals |
| `differences.csv` | Paired differences: candidate minus baseline run, ours minus each baseline arm, ours tract minus AGS |
| `results_unmatched.csv` | Each arm over every city where it alone is defined |
| `secondary_cells.csv` | Declared E4 spill cells |
| `calibration.csv` | Held-out interval coverage and log-width factors |
| `cell_scores.csv` | Per-cell scored predictions |
| `truth_completeness.csv` | Truth coverage per fold, city and offense |
| `run_manifest.json` | Run configuration, dependency fingerprint, completion time, result hashes |

The 2025.3 release run is **`gold_e3_rc3b`**: 15 spatial folds, 2 temporal folds and the
E4 model fold scored on `rc3-2025.3`. `--baseline-run gold_rc4b` re-scores the published
2025.2.1 surface's retained predictions on this run's keyed truth (`baseline_ours`; the paired
change is in `docs/CHANGELOG.md`). `--recorded-temporal-blend-run gold_rc4b` takes the temporal
folds' larceny blend tables that run recorded, accepted only where its manifest hashes the table
and every row matches this run's blend base.

```bash
uv run python scripts/eval/gold_eval.py --run-id gold_e3_rc3b \
  --score-candidate state/candidates/rc3-2025.3 --baseline-run gold_rc4b \
  --recorded-temporal-blend-run gold_rc4b
```

Fold wall time: 24,940 s.

Level-lane holdout, equivalence check and jurisdiction-total checks:

```bash
uv run python scripts/eval/level_holdout.py --year 2024 --folds 10 --mask-share 0.2 \
  --label production_default --out-dir state/eval/level_v12 \
  --cache-dir state/eval/level_v12/cache --refresh-panel --skip-remainder
uv run python scripts/eval/level_holdout.py --equivalence-check --year 2025 --panel-year-end 2025 \
  --out-dir state/eval/level_equivalence_rc3
uv run python scripts/eval/total_checks.py --year 2025 --out-dir state/eval/total_checks_2025_rc3
```

Runtimes: level holdout 781 s, equivalence check 416 s. Copy the results tables into
`docs/EVALUATION.md`.
