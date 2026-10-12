import gzip
import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts/diagnostics/verify_frontend_tile_samples.py"
SPEC = importlib.util.spec_from_file_location("verify_frontend_tile_samples", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_synthetic_mvt_round_trip_preserves_support_and_rounded_values():
    import mapbox_vector_tile

    properties = {
        "block_group_geoid": "160010001023",
        "i_evw": 29.4,
        "d_tot": 179.4567,
        "es_rob": 12.34,
        "ipt_mur": 60.1,
    }
    encoded = mapbox_vector_tile.encode(
        {
            "name": "blockgroups",
            "features": [{
                "geometry": {"type": "Point", "coordinates": [2048, 2048]},
                "properties": properties,
            }],
        }
    )
    decoded = MODULE.decoded_properties(
        gzip.decompress(gzip.compress(encoded)),
        "blockgroups",
        "block_group_geoid",
        "160010001023",
    )
    errors = MODULE.compare_properties(
        geoid="160010001023",
        lane="block_group",
        properties=decoded,
        expected={
            "i_evw": (29.44, 1),
            "d_tot": (179.45671, 4),
            "es_rob": (12.344, 2),
            "ipt_mur": (60.12, 1),
            "ip_mur": (None, 1),
        },
    )
    assert errors == []


def test_stratified_plan_is_deterministic_and_adds_parent_tracts():
    bg_columns = MODULE.parquet_columns("block_group")
    tract_columns = MODULE.parquet_columns("tract")
    bg = pd.DataFrame([{column: 1.0 for column in bg_columns} for _ in range(5)])
    bg[MODULE.BG_GEOID] = ["010010001001", "010010001002", "010010001003", "010010001004", "010010001005"]
    bg["population_2025"] = [1000, 1, 500, 0, 100]
    bg["land_area_sq_mi"] = [0.01, 100, 1, 1, 1]
    bg["special_use_tract_flag"] = 0
    bg["crime_density_total"] = [10, 1, 20, 0, 2]
    bg.loc[4, "index_robbery_primary"] = None
    bg.loc[2, "index_robbery_primary"] = 999

    tract = pd.DataFrame([{column: 1.0 for column in tract_columns} for _ in range(3)])
    tract[MODULE.TRACT_GEOID] = ["01001000100", "01001000200", "01001000300"]
    tract["index_murder_primary"] = [2, 100, 3]
    tract["index_rape_primary"] = [2, 3, 200]
    tract["index_harm_burden_resident"] = [300, 2, 3]

    first = MODULE.choose_rows(bg, tract)
    second = MODULE.choose_rows(bg.sample(frac=1, random_state=4), tract.sample(frac=1, random_state=5))
    assert first == second
    tags = {tag for item in first for tag in item["tag"].split(",")}
    assert {"ordinary_urban", "ordinary_rural", "regular_primary_extreme", "valid_zero", "regular_no_data"} <= tags
    assert any("parent_of_" in item["tag"] for item in first)
