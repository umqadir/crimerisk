#!/usr/bin/env bash
# 02 - Ensure the block-group geometry + water mask are present.
#
# The choropleth is drawn at CENSUS BLOCK GROUP granularity (~238k polygons,
# finer than tract). 03 joins the index table onto the full-resolution 2020
# TIGER/Line block-group shapefiles, one zip per state, at:
#     data/tiger_bg/tl_2020_{ss}_bg.zip
# These per-state files are committed to the repo, so there is normally nothing
# to download here — this step just VERIFIES they exist and fetches the small
# Natural Earth ocean polygon that 03 uses to clip coastal block groups that
# bleed into open water.
#
# Run:  bash frontend/build/02_fetch_geometry.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$REPO/frontend/tmp"
BG_DIR="$REPO/data/tiger_bg"
TRACT_DIR="$REPO/data/tiger_tracts"
mkdir -p "$TMP"

# Verify the per-state block-group TIGER files are present. 03 reads them by
# state FIPS (tl_2020_{ss}_bg.zip) for every state in the index data.
if [ -d "$BG_DIR" ] && ls "$BG_DIR"/tl_2020_*_bg.zip >/dev/null 2>&1; then
  n=$(ls "$BG_DIR"/tl_2020_*_bg.zip | wc -l | tr -d ' ')
  echo "Block-group TIGER geometry present: $n state files in $BG_DIR."
else
  echo "ERROR: no block-group TIGER files found in $BG_DIR" >&2
  echo "       expected per-state files like tl_2020_06_bg.zip." >&2
  exit 1
fi

# Verify the per-state tract TIGER files are present. The murder/rape layers are
# drawn at census-tract granularity, so 03 also reads the tract geometry by state
# FIPS (tl_2020_{ss}_tract.zip). The 2020 vintage matches the published index
# GEOIDs everywhere, including Connecticut's legacy 2020 county-based GEOIDs (the
# same reasoning as the block-group chain); any non-2020 vintage present is ignored.
if [ -d "$TRACT_DIR" ] && ls "$TRACT_DIR"/tl_2020_*_tract.zip >/dev/null 2>&1; then
  n=$(ls "$TRACT_DIR"/tl_2020_*_tract.zip | wc -l | tr -d ' ')
  echo "Tract TIGER geometry present: $n state files in $TRACT_DIR."
else
  echo "ERROR: no tract TIGER files found in $TRACT_DIR" >&2
  echo "       expected per-state files like tl_2020_06_tract.zip." >&2
  exit 1
fi

# Natural Earth 10m ocean — the water mask used in 03 to clip block groups that
# bleed into oceans / harbors / tidal estuaries (the Hudson and East River are
# inside this polygon). Tiny (~3 MB). 03 erases an inward-buffered version of
# this from water-heavy block groups so coastal cells stop extending into water.
NE_DIR="$TMP/ne"
NE_ZIP="$NE_DIR/ne_10m_ocean.zip"
NE_URL="https://naciscdn.org/naturalearth/10m/physical/ne_10m_ocean.zip"
if [ -f "$NE_DIR/ne_10m_ocean.shp" ]; then
  echo "Natural Earth ocean already present at $NE_DIR — skipping."
else
  mkdir -p "$NE_DIR"
  echo "Fetching Natural Earth 10m ocean ..."
  if curl -fsSL --retry 3 -o "$NE_ZIP" "$NE_URL"; then
    unzip -oq "$NE_ZIP" -d "$NE_DIR"
    echo "Extracted NE ocean to $NE_DIR."
  else
    echo "WARNING: could not download NE ocean; 03 will skip the water clip." >&2
  fi
fi
