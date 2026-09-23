# Public map build

```
uv run python frontend/build/01_extract.py      # public surfaces + county rollups + benchmarks
uv run python frontend/build/02_geometry.py     # TIGER 2020 join -> GeoJSONSeq per geography
bash            frontend/build/03_tiles.sh all  # three PMTiles archives
uv run python frontend/build/04_shards.py       # GEOID lookup shards + per-state benchmarks
uv run python frontend/build/05_download.py     # public CSV / Parquet / GeoParquet package
uv run python frontend/build/06_stage_site.py   # bootstrap manifest + staged site
uv run python frontend/serve.py 8777 "$CRIMERISK_TILES_ROOT/dist/site"
```

Field contract: `frontend/build/crschema.py`.
Source parquet and data year: `frontend/build/snapshot_config.env`.
Generated artifacts root: `CRIMERISK_TILES_ROOT`, default `../crimerisk-tiles`
relative to the repository.

| Layer | Geography | Zooms | Source |
|---|---|---|---|
| counties | 3,108 counties | 3-7 | population-weighted tract rollup, display only |
| tracts | 83,776 tracts | 5-12 | published tract surface |
| blockgroups | 238,193 block groups | 8-12 | published block-group surface |

Tiles carry 25 attributes per feature. Everything else the card shows comes
from the GEOID-keyed lookup shards.
