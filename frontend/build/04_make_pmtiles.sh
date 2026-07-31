#!/usr/bin/env bash
# 04 - Build vector tiles (PMTiles) from the joined GeoJSONSeq.
#
# PMTiles is a single-file tile archive served by a static HTTP server with
# range requests; MapLibre reads it directly. This is what keeps ~238k block
# groups smooth — the browser only fetches tiles for the current view.
#
# Run:  bash frontend/build/04_make_pmtiles.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Snapshot data year + published-artifacts directory: single point of control in
# snapshot_config.env (see 01). Both tile archives are written to OUT_DIR.
# shellcheck source=frontend/build/snapshot_config.env
source "$REPO/frontend/build/snapshot_config.env"
OUT_DIR="$REPO/${CRIMERISK_SNAPSHOT_OUT:-frontend/public}"
SRC="$REPO/frontend/tmp/block_groups_with_indices.geojsonl"
OUT="${OUT_DIR}/crimerisk_block_groups_${CRIMERISK_SNAPSHOT_YEAR}.pmtiles"
# Tract lane: murder/rape publish at census-tract support and render from a second
# tile archive built from tract geometry.
TRACT_SRC="$REPO/frontend/tmp/tracts_with_indices.geojsonl"
TRACT_OUT="${OUT_DIR}/crimerisk_tracts_${CRIMERISK_SNAPSHOT_YEAR}.pmtiles"

if [ ! -f "$SRC" ]; then
  echo "ERROR: $SRC not found. Run 03_join_geometry.py first." >&2
  exit 1
fi
if [ ! -f "$TRACT_SRC" ]; then
  echo "ERROR: $TRACT_SRC not found. Run 03_join_geometry.py first." >&2
  exit 1
fi

# Staleness guard. 01 owns the attribute encodings (notably the density field
# precision, which the viewer's paint floor is derived from); 03 is what bakes them
# into the features this step tiles. If 01 has re-run since 03, this GeoJSONSeq is
# from the previous encoding and tiling it would ship attributes that disagree with
# the snapshot 05 is about to stamp — silently, since the tiles look fine.
STATS="$REPO/frontend/tmp/index_stats.json"
for f in "$SRC" "$TRACT_SRC"; do
  if [ -f "$STATS" ] && [ "$f" -ot "$STATS" ]; then
    echo "ERROR: $(basename "$f") is older than index_stats.json — 01 has re-run since 03." >&2
    echo "       Re-run 03_join_geometry.py before tiling (its attribute encoding is stale)." >&2
    exit 1
  fi
done
mkdir -p "$OUT_DIR"

echo "Building PMTiles from $(basename "$SRC") ..."

# Goal: smooth national view AND crisp neighborhood-level zoom (every block group
# visible, boundaries smooth, adjacent cells flush — no slivers/gaps), in one
# archive. Block groups are FINER than tracts (~238k vs ~84k polygons, many tiny
# urban cells), so the priorities are: a high enough max zoom to resolve small
# cells, and NO coalescing that would merge those small cells away.
#
# -Z3 -z12       : zoom range 3 (national) .. 12 (neighborhood). Block groups are
#                  small; z12 gives enough tile resolution that even compact urban
#                  block groups survive as distinct polygons. The source is
#                  full-resolution TIGER geometry (dense vertices), so the high max
#                  zoom lets that detail survive instead of being thrown away.
# --detect-shared-borders : THE GAP FIX. Without this, tippecanoe simplifies each
#                  polygon's edges independently, so a border shared by two cells
#                  gets simplified differently on each side -> slivers/gaps appear
#                  between neighbors. This flag simplifies shared edges identically
#                  on both sides, keeping adjacent block groups flush.
# --no-simplification-of-shared-nodes : never move a vertex shared between features,
#                  so junctions where 3+ block groups meet stay pinned.
# --simplification=2 : light simplification only (closer to source vertices).
# --no-tiny-polygon-reduction : keep small/thin block groups as real polygons
#                  instead of collapsing them to dots — critical at BG granularity,
#                  where many genuine urban cells are tiny.
# -rg / --drop-densest-as-needed : at LOW zooms only, thin the densest tiles so the
#                  national view stays light; full detail returns as you zoom in.
#                  (At z12 the per-tile cap is removed below, so nothing is dropped
#                  at neighborhood zoom.)
# NO --coalesce-*-as-needed : we deliberately do NOT coalesce small features. At BG
#                  granularity coalescing would merge real, distinct small urban
#                  block groups into their neighbors — the opposite of the point of
#                  switching to block groups.
# --no-feature-limit / --no-tile-size-limit : never silently drop a block group for
#                  exceeding the per-tile feature cap (we want every BG at max zoom).
# Short attribute keys (set in 03) keep per-feature payload tiny.
tippecanoe \
  -o "$OUT" \
  --force \
  -l blockgroups \
  -n "CrimeRisk block groups ${CRIMERISK_SNAPSHOT_YEAR}" \
  -Z3 -z12 \
  --detect-shared-borders \
  --no-simplification-of-shared-nodes \
  --simplification=2 \
  --no-tiny-polygon-reduction \
  --drop-densest-as-needed \
  --no-feature-limit \
  --no-tile-size-limit \
  "$SRC"

# Tract tiles (murder/rape layers). Tracts are coarser than block groups (~84k vs
# ~238k polygons), so the same zoom range and simplification flags are comfortably
# sufficient; the shared-border handling matters just as much (adjacent tracts must
# stay flush, junctions pinned) so the identical flag set is used.
echo
echo "Building tract PMTiles from $(basename "$TRACT_SRC") ..."
tippecanoe \
  -o "$TRACT_OUT" \
  --force \
  -l tracts \
  -n "CrimeRisk tracts ${CRIMERISK_SNAPSHOT_YEAR}" \
  -Z3 -z12 \
  --detect-shared-borders \
  --no-simplification-of-shared-nodes \
  --simplification=2 \
  --no-tiny-polygon-reduction \
  --drop-densest-as-needed \
  --no-feature-limit \
  --no-tile-size-limit \
  "$TRACT_SRC"

echo
echo "Done. PMTiles written:"
ls -lh "$OUT" "$TRACT_OUT"
echo
echo "Block-group tile archive header:"
pmtiles show "$OUT" 2>/dev/null | head -25 || true
echo
echo "Tract tile archive header:"
pmtiles show "$TRACT_OUT" 2>/dev/null | head -25 || true
