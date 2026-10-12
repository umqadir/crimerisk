"""Download the BIA AIAN National Land Area Representation (LAR) polygons.

    uv run python scripts/pull/pull_bia_lar.py

| Output | Content |
|---|---|
| data/bia_lar/bia_aian_national_lar.geojson | every LAR feature, all attributes, EPSG:4269 coordinates as GeoJSON |
| data/bia_lar/layer_metadata.json, layer_metadata.xml | the layer's service description and ISO metadata |

Source: https://biamaps.geoplatform.gov/server/rest/services/DivLTR/BIA_AIAN_National_LAR/FeatureServer/0
(paged by OBJECTID, 50 features a page). GISACRES is computed by BIA in Web Mercator; areas
are recomputed in an equal-area projection where they are used.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
LAYER = (
    "https://biamaps.geoplatform.gov/server/rest/services/DivLTR/BIA_AIAN_National_LAR/FeatureServer/0"
)
OUT = REPO_ROOT / "data" / "bia_lar"
PAGE = 50


def _get(url: str) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": "crimerisk-pull/1.0"}), timeout=600) as response:
        return response.read()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layer_metadata.json").write_bytes(_get(f"{LAYER}?f=pjson"))
    (OUT / "layer_metadata.xml").write_bytes(_get(f"{LAYER}/metadata"))
    count = json.loads(_get(f"{LAYER}/query?where=1%3D1&returnCountOnly=true&f=json"))["count"]
    features: list[dict] = []
    for offset in range(0, int(count), PAGE):
        page = json.loads(
            _get(
                f"{LAYER}/query?where=1%3D1&outFields=*&outSR=4269&returnGeometry=true"
                f"&orderByFields=OBJECTID&resultOffset={offset}&resultRecordCount={PAGE}&f=geojson"
            )
        )
        features.extend(page["features"])
    ids = {feature["properties"]["OBJECTID"] for feature in features}
    if len(features) != int(count) or len(ids) != int(count):
        raise RuntimeError(f"expected {count} LAR features, got {len(features)} ({len(ids)} distinct)")
    (OUT / "bia_aian_national_lar.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
    )
    print(f"{OUT.relative_to(REPO_ROOT)}/bia_aian_national_lar.geojson: {len(features)} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
