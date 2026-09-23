# Build

Edition 2025.1. Every command runs from the repository root.

```bash
uv sync
```

## Inputs

Raw inputs are not in this repository. They are staged under `data/` and `state/`
before the first stage runs.

| Input | Source | Staged at |
|---|---|---|
| FBI UCR Return A master file, 2025 | FBI UCR | `data/FBI-UCR-Return-A-Parsed-2025-20260921/return_a_master_annual_2025.parquet` |
| FBI CIUS annual tables | FBI | `data/FBI-CIUS-Annual/<year>/raw/` |
| FBI CDE state estimates | FBI | `data/FBI-CDE-Estimates-1979-2024/estimated_crimes_1979_2024.csv` |
| FBI NIBRS tables, 2024 | FBI | `data/FBI-NIBRS-Tables-2024/raw/statesAndFederal.zip` |
| NIBRS segment files, Kaplan | ICPSR | `data/NIBRS-Kaplan-1991-2024/` |
| SRS offenses known, Kaplan | ICPSR | `data/SRS-Kaplan-1960-2024/` |
| State publications | FDLE, MS TOPS, NY DCJS, CT DESPP | `data/FDLE-FIBRS-2024/`, `data/MS-TOPS-2024/`, `data/NY-DCJS-2024/`, `data/CT-DESPP-2024/` |
| City incident feeds | City open-data portals | `data/city-incidents/<city>/` |
| TIGER/Line 2020 | Census Bureau | `data/tiger_tabblock20/`, `data/tiger_bg/`, `data/tiger_tracts/`, `data/tiger_counties/`, `data/tiger_places/`, `data/tiger_cousub/` |
| Population estimates | Census Bureau | `data/Census-PopEst-2020-2025/co-est2025-alldata.csv` |
| ACS 5-year 2020 to 2024 | Census Bureau | `data/ACS-5yr-2020-2024/parsed/` |
| LEHD LODES 2023 | Census Bureau | `data/LODES/parsed/` |
| BLS QCEW 2024 | BLS | `data/qcew/raw/` |
| LandScan USA 2021 | Oak Ridge National Laboratory | `data/LandScan-USA/` |
| Overture places | Overture Maps, release resolved at pull time | `data/Overture-Places/` |
| NLCD 2023 | USGS | `data/NLCD/parsed/` |
| Roads, HPMS, NCES EDGE, CMS hospitals | FHWA, NCES, CMS | `data/roads/`, `data/HPMS/`, `data/NCES-EDGE/`, `data/CMS-Hospital-General-Info/` |

`src/crimerisk/required_inputs.py` holds the expected path and presence check for
each input. Source notices are in `ATTRIBUTION.md`.

## Stage order

Run the stages in this order. Each command writes into `state/` and is reused by the
next stage unless a `--force-*` flag is passed.

| # | Stage | Command | Runtime |
|---|---|---|---|
| 1 | Input manifest | `uv run python main.py build-input-manifest` | 1 s |
| 2 | Agency master | `uv run python main.py build-agency-master` | 5 s |
| 3 | Reference layers | `uv run python main.py build-reference-layers` | 13 s |
| 4 | Geometry and crosswalks | `uv run python main.py build-geometry` | 24 s |
| 5 | Observations panel | `uv run python main.py build-observations --year-start 2018 --year-end 2025` | 2m 18s |
| 6 | Reporting regimes | `uv run python main.py build-reporting-regimes --year-start 2018 --year-end 2025` | 5m 26s |
| 7 | City incident shares | `uv run python main.py build-city-incident-shares --year-start 2018 --year-end 2025` | 6 s |
| 8 | Accounting controls | `uv run python main.py build-controls --year 2025 --enable-imputation-v2` | 4m 54s |
| 9 | Smoothed controls | `uv run python main.py build-smoothed-controls --year 2025` | 1 s |
| 10 | Exposure normalizers | `uv run python main.py build-exposure-normalizers --year 2025` | 9 s |
| 11 | Mixture experts | `uv run python main.py build-mixture-experts --year 2025 --enable-soft-shrinkage` | 13 s |
| 12 | Outputs | `uv run python main.py build-outputs --year 2025 --candidate-run <run-id>` | 6m 28s |

Stage 12 runs allocation, composites, the uncertainty layer and the output writer in
one process. It writes to `state/candidates/<run-id>/`.

Runtimes are wall time from the 2025.1 assembly run on an Apple silicon laptop, with
`state/` already carrying the staged artifacts. A stage whose dependency stamp still
matches reuses its artifact and returns in seconds; stages 5, 6, 8 and 12 rebuild
every time and their figures are full rebuilds.

Running this order again is a no-op on the artifacts. Stages 5, 6 and 8 redo their
work and land on the same bytes; every other stage reports its artifact current and
returns in about a second. Measured on the 2025.1 assembly: a second pass over stages
1 to 11 left all 218 parquet and JSON artifacts under `state/modeling`,
`state/controls`, `state/geometry`, `state/reference` and `state/cache/mixture`
bit-identical, with stages 7, 10 and 11 dropping to 6 s, 1 s and 1 s. A stage whose
inputs did change costs its full rebuild instead: stage 7 takes 1m 17s when the feed
inputs move, and stages 10 and 11 take the figures in the table above.

Nothing in 2025.1 is pinned, so a checkout of this commit with the staged inputs
reproduces the release from stage 1.

The `--enable-imputation-v2` and `--enable-soft-shrinkage` flags on stages 8 and 11
must match the flags stage 12 is run with, or stage 12 fails on a mode mismatch.

The exposure normalizers come before the mixture experts because the mixture's
exposure expert is defined over the normalizer surface and reads it: with the
normalizer artifact absent the expert build stops with a `FileNotFoundError`, and a
normalizer surface rebuilt afterwards leaves the expert table stale. Nothing else in
the order is negotiable either; each stage's stamp names the artifacts of the ones
above it.

A stage decides whether to reuse its artifact from the dependency stamp beside it
(`<artifact>.deps.json`), which records the content digest of every input. The
comparison is by digest alone, so a stamp written in one checkout is read correctly
in another: a worktree, a clone and the isolated runtime are the same tree at
different absolute paths, and moving a tree is not a changed input.

## Outputs

| File | Contents |
|---|---|
| `state/candidates/<run-id>/crimerisk_block_group_2025_ags_core.parquet` | Block-group surface |
| `state/candidates/<run-id>/crimerisk_tract_2025_ags_core.parquet` | Tract surface |
| `state/candidates/<run-id>/manifest.json` | Resolved argv, source SHA, input mtimes |
| `state/candidates/<run-id>/validation_summary.json` | Validator result |
| `state/candidates/<run-id>/unlocated_mass_2025.parquet` | Mass that could not be placed on a footprint |

## Validator

```bash
uv run python scripts/diagnostics/validate_release_outputs.py --year 2025 \
  --state-output-dir state/candidates/<run-id> \
  --summary-out state/candidates/<run-id>/validation_summary.json
```

The gate exits non-zero on any issue. Its blocking blocks are total-lane integrity,
allocation coherence, index coherence, spatial artifacts, the feature and redlining
audit, frontend readiness, and the release evaluation's presence and candidate
binding. Promotion to `state/output/` requires a green gate.

`_check_release_evaluation`:

| Requirement | 2025.1.1 |
|---|---|
| Results and manifest for `RELEASE_GOLD_RUN_ID` present; run id and build year match | Blocking |
| Evaluated candidate equals the candidate `frontend/build/snapshot_config.env` publishes | Blocking |
| Spatial TVD: `ours` below the population arm for every offense except rape | Reported in `release_evaluation.failures`, not blocking (`RELEASE_GOLD_TVD_BLOCKING = False`) |

Rape is reported and not gated: one eligible fold city.

## Tiles and site

`frontend/build/snapshot_config.env` names the candidate parquets the snapshot is
frozen from. Set it before running these steps. Large artifacts are written to
`CRIMERISK_TILES_ROOT`, not into the repository.

| # | Step | Command | Runtime |
|---|---|---|---|
| 1 | Public surfaces and benchmarks | `uv run python frontend/build/01_extract.py` | 3 s |
| 2 | Geometry join, GeoJSONSeq | `uv run python frontend/build/02_geometry.py` | 40 s |
| 3 | PMTiles archives | `bash frontend/build/03_tiles.sh` | 4m 59s |
| 4 | Lookup shards and benchmarks | `uv run python frontend/build/04_shards.py` | 8 s |
| 5 | Download package | `uv run python frontend/build/05_download.py` | 45 s |
| 6 | Stage the site | `uv run python frontend/build/06_stage_site.py` | 5 s |
| 7 | Browser verification | `uv run --with playwright python frontend/build/07_verify.py` | 51 s |

Runtimes are the 2025.1 tile build, same machine. Step 7 needs Playwright and its
Chromium build in the running interpreter; it is not a project dependency, which is
why it is run through `uv run --with playwright`. Install the browser once with
`playwright install chromium`.

Step 6 writes the public `https://tiles.qqlab.io/2025.1/` URLs into the staged site.
Step 7 measures the map against a local range server, so stage the site twice: once
with `CRIMERISK_ASSET_BASE_URL=http://127.0.0.1:8791/data/tiles` for step 7, then
again with no override to produce the tree that is uploaded.

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
uv run python scripts/eval/gold_eval.py --run-id <eval-run-id>
```

`--dry-run` prints the fold list and the build command for each fold without building.
Folds are selected with repeated `--folds` arguments from `spatial`, `temporal` and
`model`; all three run by default. Spatial folds rebuild one surface per feed city
with `--exclude-feed-city`; `--cities` restricts that set to named feed-city keys.
Temporal folds rebuild with `--feed-year-end 2021` and `--feed-year-end 2023`.
Bootstrap resampling is clustered by city, 1,000 iterations by default, settable with
`--bootstrap-iterations`. Fold surfaces are deleted after scoring.

Every fold builds a deliberately held-out city incident feed, so it is given its own
`--feed-inputs-dir` under `state/eval/<eval-run-id>/inputs/<fold-id>/`. The share
surface, the per-city reconciliation tables and the combined reconciliation table for
that fold are written there, never into `state/modeling`. At the end of the run the
harness re-hashes the shared city-feed artifacts it read and fails if any changed.

Results are written to `state/eval/<eval-run-id>/`:

| File | Contents |
|---|---|
| `results.csv`, `results.md` | One row per offense by fold type, with baselines and intervals |
| `calibration.csv` | Held-out interval coverage and log-width factors |
| `cell_scores.csv` | Per-cell scored predictions |
| `fold_record.json` | Per-fold build manifest and artifact hashes |
| `run_manifest.json` | Run configuration, completion time, result hashes |

Runtime is 3h 37m 11.2s for the full 16-fold set (`gold_v1`) and about 50 minutes for
the 2025.1 release set: five feed-free spatial folds plus two temporal folds. The
release run is **`gold_v53r2`**, the one tabled in `docs/EVALUATION.md`. `gold_v53` is
the same seven folds scored before the stage-10 normalizer surface and the stage-11
expert table were retrained; it is kept as the previous candidate.

```bash
uv run python scripts/eval/gold_eval.py --run-id gold_v53r2 \
  --folds spatial --folds temporal \
  --cities san_francisco,washington_dc,denver,minneapolis,st_louis_mo
```

Copy the results tables into `docs/EVALUATION.md`.
