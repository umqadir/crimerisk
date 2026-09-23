#!/usr/bin/env bash
# 03 - Build one PMTiles archive per geography, each with its own zoom range.
#
#   counties     z3-7    national/state overview; population-weighted rollup.
#                        The map paints it alone below z5 and keeps it as a
#                        backfill under the tract layer at z5-7, so a tract
#                        thinned out of a dense overview tile reveals a county
#                        average rather than a hole.
#   tracts       z5-12   regional view, plus murder/rape and the block-group
#                        null fallback at neighborhood zoom
#   blockgroups  z8-12   neighborhood view
#
# --maximum-tile-bytes=250000 is the p95 budget. It binds only on the handful
# of dense overview tiles that exceed it; every other tile is untouched. A
# feature dropped there falls back to the coarser layer underneath.
#
# Tile-size limits are ENFORCED (no --no-tile-size-limit / --no-feature-limit).
# Overview layers may simplify hard and drop tiny polygons; the neighborhood
# layer keeps every block group and relies on shared-border detection so
# adjacent cells stay flush.
#
# Each archive is built to .mbtiles first so per-zoom tile-byte statistics can
# be read straight out of SQLite, then converted to .pmtiles.
#
# Run:  bash frontend/build/03_tiles.sh [counties|tracts|blockgroups]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/frontend/build/snapshot_config.env"
# Same default as crschema.py: a sibling of the repository.
TILES_ROOT="${CRIMERISK_TILES_ROOT:-$(dirname "$REPO")/crimerisk-tiles}"
WORK="$TILES_ROOT/work"
DIST="$TILES_ROOT/dist"
LOGS="$TILES_ROOT/logs"
YEAR="$CRIMERISK_SNAPSHOT_YEAR"
mkdir -p "$DIST" "$LOGS"

SHARED="-q --detect-shared-borders --no-simplification-of-shared-nodes --drop-densest-as-needed --hilbert"

build_counties() {
  tippecanoe -o "$WORK/counties.mbtiles" --force \
    -l counties -n "CrimeRisk county averages $YEAR" \
    -Z3 -z7 --simplification=10 --maximum-tile-bytes=250000 \
    $SHARED \
    "$WORK/geo_county.geojsonl"
}

build_tracts() {
  tippecanoe -o "$WORK/tracts.mbtiles" --force \
    -l tracts -n "CrimeRisk census tracts $YEAR" \
    -Z5 -z12 --simplification=6 --maximum-tile-bytes=250000 \
    $SHARED \
    "$WORK/geo_tract.geojsonl"
}

build_blockgroups() {
  tippecanoe -o "$WORK/blockgroups.mbtiles" --force \
    -l blockgroups -n "CrimeRisk block groups $YEAR" \
    -Z8 -z12 --simplification=3 --maximum-tile-bytes=250000 --no-tiny-polygon-reduction \
    $SHARED \
    "$WORK/geo_bg.geojsonl"
}

convert() {
  local name="$1"
  rm -f "$DIST/crimerisk_${name}_${YEAR}.pmtiles"
  pmtiles convert "$WORK/${name}.mbtiles" "$DIST/crimerisk_${name}_${YEAR}.pmtiles"
  ls -l "$DIST/crimerisk_${name}_${YEAR}.pmtiles"
}

target="${1:-all}"
case "$target" in
  counties)    build_counties    2>&1 | tee "$LOGS/tippecanoe_counties.log";    convert counties ;;
  tracts)      build_tracts      2>&1 | tee "$LOGS/tippecanoe_tracts.log";      convert tracts ;;
  blockgroups) build_blockgroups 2>&1 | tee "$LOGS/tippecanoe_blockgroups.log"; convert blockgroups ;;
  all)
    build_counties    2>&1 | tee "$LOGS/tippecanoe_counties.log"
    build_tracts      2>&1 | tee "$LOGS/tippecanoe_tracts.log"
    build_blockgroups 2>&1 | tee "$LOGS/tippecanoe_blockgroups.log"
    convert counties; convert tracts; convert blockgroups
    ;;
  *) echo "unknown target: $target" >&2; exit 1 ;;
esac
