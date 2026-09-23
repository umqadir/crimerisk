# Changelog

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
