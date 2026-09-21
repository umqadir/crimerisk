# Changelog

## 2025.1

Data year 2025. Previous release: 2026-08-18.

- The level lane consumes the FBI 2025 Return A master file.
- Public measures reduced from nine to two: Crime exposure and Per resident.
- Density, harm-weighted burden and multi-offense relative scores left the public product.
- Composites are Overall, Violent and Property, as national expected-count-share-weighted averages of the offense indexes.
- A composite is suppressed when any component offense index is suppressed.
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
- Nothing in the build is pinned. The stage-10 expert table and stage-11 normalizer surface, which 2025.1 previously shipped as reviewed 2026-08-18 artifacts, are retrained from current inputs; the release is reproducible from the documented stage order.
- The documented stage order now builds the exposure normalizers before the mixture experts. The mixture's exposure expert reads the normalizer surface, so the old order failed on a fresh build and left a stale expert table on an incremental one.
- Retraining the mixture experts against the refreshed crosswalk and jurisdiction estimates moved the feature panel from 10,863 to 10,881 jurisdictions and the training rows from 8,282 to 8,286. National anchors, published cell counts and the published universe are unchanged; block-group indexes move by under 20 index points at p99 on the composites.
- The exposure normalizer surface retrains byte-identically, and its summary now records the residential leg it was built with.
- The person-exposure residential leg is LandScan nighttime population. The Census-residential arm (`--residential-leg-source census_release_population`) stays an opt-in sensitivity arm under its own artifact and normalizer version and is not published.
