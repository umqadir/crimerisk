import importlib.util
import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BUILD = Path(__file__).parents[1] / "frontend" / "build"
sys.path.insert(0, str(BUILD))
SPEC = importlib.util.spec_from_file_location("evidence_bundle", BUILD / "05c_evidence_bundle.py")
BUNDLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUNDLE)


def test_ags_token_matches_ags_fields_but_not_this_projects_ags_core_names():
    assert BUNDLE.is_ags("prediction_ags_2022a_tract")
    assert BUNDLE.is_ags("ags_2022a_tract")
    assert BUNDLE.is_ags("AGS")
    assert not BUNDLE.is_ags("reuse_flags")
    assert not BUNDLE.is_ags("block_group_ags_core")
    assert not BUNDLE.is_ags("crimerisk_block_group_2025_ags_core.parquet")


def test_csv_drops_ags_value_columns_keeps_flags_and_every_other_byte():
    raw = (
        "city,ags_uses_city_feed,prediction_ags_2022a_tract,tvd\n"
        "Denver,False,12.5,0.1234567890123\n"
        "Austin,True,3.0,0.2\n"
    ).encode()
    out, dropped = BUNDLE.strip_csv(raw)
    assert dropped == ["prediction_ags_2022a_tract"]
    assert out.decode() == (
        "city,ags_uses_city_feed,tvd\n"
        "Denver,False,0.1234567890123\n"
        "Austin,True,0.2\n"
    )


def test_parquet_drops_ags_value_columns_and_keeps_boolean_flags():
    buf = io.BytesIO()
    pq.write_table(pa.table({
        "cell": ["a", "b"],
        "ags_uses_city_feed": [True, False],
        "prediction_ags_2022a_tract": [1.0, 2.0],
    }), buf)
    out, dropped = BUNDLE.strip_parquet(buf.getvalue())
    assert dropped == ["prediction_ags_2022a_tract"]
    assert pq.read_table(io.BytesIO(out)).column_names == ["cell", "ags_uses_city_feed"]


def test_json_drops_ags_keys_at_any_depth():
    raw = json.dumps({
        "inputs": {"ags_2022a_tract": {"path": "x", "sha256": "y"}, "truth": {"path": "t"}},
        "summary": {"ags_values_used": False, "block_group_ags_core": {"rows": 1}},
    }).encode()
    out, dropped = BUNDLE.strip_json(raw)
    assert dropped == ["/inputs/ags_2022a_tract"]
    assert json.loads(out) == {
        "inputs": {"truth": {"path": "t"}},
        "summary": {"ags_values_used": False, "block_group_ags_core": {"rows": 1}},
    }
