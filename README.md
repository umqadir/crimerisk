# CrimeRisk

A neighborhood crime index built from public data. It covers the seven FBI Part I
offenses at census block group and census tract, for the 48 contiguous states and
the District of Columbia. 100 is the US average for the selected offense and measure.

## The two measures

**Crime exposure** divides recorded offenses by what each offense happens to: people
present for violent offenses, premises for burglary, vehicles for motor vehicle theft.
**Per resident** divides the same offenses by resident population.

Crime exposure is the default measure. The map card shows both values.

## Coverage

| Field | Value |
|---|---|
| Edition | 2025.3 |
| Data year | 2025 |
| History window | 2018 to 2025 |
| Offenses | Murder, rape, robbery, aggravated assault, burglary, larceny, motor vehicle theft |
| Composites | Overall, Violent, Property |
| Block-group support | Robbery, aggravated assault, burglary, larceny, motor vehicle theft |
| Tract support | Murder, rape |
| Boundaries | 2020 Census |
| Block groups published | 238,193 |
| Tracts published | 83,776 |
| Block groups with a published per-resident index | 233,612 (300 residents or more in ordinary areas, 550 in special-use areas) |
| Tracts with a published per-resident index | 82,797 |
| Block groups with a published crime-exposure index | 234,673 (aggravated assault) to 236,733 (larceny) |
| Tracts with a published murder and rape crime-exposure index | 82,742 and 82,743 |
| Block groups allocated from city incident feeds | 15 feed cities; 7.1% of block groups on larceny |
| Geography excluded | Alaska, Hawaii, territories |

## Map

https://umqadir.github.io/crimerisk-map/

## Download

One table per geography, nationwide and per state. The schema is 64 columns at block
group and 63 at tract. See [docs/FIELDS.md](docs/FIELDS.md).

Download page, including per-state tables and checksums:
https://umqadir.github.io/crimerisk-map/download.html. The object host serves files
only; `https://tiles.qqlab.io/2025.3/downloads/` has no listing.

| File | Link |
|---|---|
| **Download page (all files)** | https://umqadir.github.io/crimerisk-map/download.html |
| Block group, CSV | https://tiles.qqlab.io/2025.3/downloads/crimerisk_2025_block_group.csv.gz |
| Block group, Parquet | https://tiles.qqlab.io/2025.3/downloads/crimerisk_2025_block_group.parquet |
| Block group, GeoParquet | https://tiles.qqlab.io/2025.3/downloads/crimerisk_2025_block_group.geoparquet |
| Tract, CSV | https://tiles.qqlab.io/2025.3/downloads/crimerisk_2025_tract.csv.gz |
| Tract, Parquet | https://tiles.qqlab.io/2025.3/downloads/crimerisk_2025_tract.parquet |
| Tract, GeoParquet | https://tiles.qqlab.io/2025.3/downloads/crimerisk_2025_tract.geoparquet |
| Per state | https://umqadir.github.io/crimerisk-map/download.html |
| Field list | https://tiles.qqlab.io/2025.3/downloads/fields_block_group.csv |
| Checksums | https://tiles.qqlab.io/2025.3/downloads/checksums.sha256 |

No incident-level record is published.

## Documentation

| Document | Contents |
|---|---|
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Estimand, sources, jurisdiction totals, allocation, denominators, limitations |
| [docs/BUILD.md](docs/BUILD.md) | Inputs, stage order, commands, runtimes, validator |
| [docs/FIELDS.md](docs/FIELDS.md) | One row per published column |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Held-out evaluation protocol and results |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Changes by edition |
| [ATTRIBUTION.md](ATTRIBUTION.md) | Data sources, licenses, required notices |

## Build

```bash
uv sync
uv run python main.py build-outputs --year 2025 --candidate-run <run-id>
uv run python scripts/diagnostics/validate_release_outputs.py --year 2025 \
  --state-output-dir state/candidates/<run-id>
uv run python frontend/build/01_extract.py
uv run python frontend/build/02_geometry.py
bash frontend/build/03_tiles.sh
```

The full stage list, including geometry, tiles, shards, downloads and site staging,
is in [docs/BUILD.md](docs/BUILD.md). Raw FBI, Census, ACS, LODES, TIGER, Overture,
NLCD, LandScan, state and city inputs are not in this repository.

From edition 2025.2, judgment steps in data onboarding and anomaly adjudication are
drafted and reviewed by Claude agents defined in `.claude/`, run through `scripts/agents/`, and
recorded with evidence, model and approval in `configs/ledger/decisions.csv`. The build reads only
ledger-approved rows, and accuracy claims come from `scripts/eval/gold_eval.py`.

## License and attribution

Source data, licenses and required source notices are listed in
[ATTRIBUTION.md](ATTRIBUTION.md). Cite as: CrimeRisk 2025 neighborhood crime index,
block group and tract, 48 states and DC.
