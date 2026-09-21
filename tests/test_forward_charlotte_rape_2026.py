from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "diagnostics"
    / "forward_charlotte_rape_2026.py"
)
spec = importlib.util.spec_from_file_location("forward_charlotte_rape_2026", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_mocked_pagination_dedupe_whole_tract_zeros_and_exact_scores(
    tmp_path, monkeypatch
):
    class Resp:
        def __init__(self, b):
            self.b = b

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return self.b

    payloads = [
        b'{"name":"CMPD Incidents"}',
        json.dumps(
            {
                "features": [{"attributes": {"OBJECTID": 1}}],
                "exceededTransferLimit": True,
            }
        ).encode(),
        json.dumps({"features": []}).encode(),
    ]
    monkeypatch.setattr(m, "urlopen", lambda *_args, **_kwargs: Resp(payloads.pop(0)))
    pages, meta = m.acquire_pages(tmp_path / "raw", page_size=1, max_pages=3)
    assert [p["offset"] for p in pages] == [0, 1] and Path(meta["path"]).exists()
    assert pages[0]["query"]["orderByFields"] == "OBJECTID ASC"
    assert (
        "DATE_REPORTED >= timestamp '2026-01-01 00:00:00'" in pages[0]["query"]["where"]
    )
    # Candidate universe: tract 001 fully Charlotte, tract 002 split and therefore excluded.
    bg = tmp_path / "bg.parquet"
    pd.DataFrame(
        {
            "block_group_geoid": [
                "370010001001",
                "370010001002",
                "370010002001",
                "370010002002",
            ],
            "tract_id": ["37001000100"] * 2 + ["37001000200"] * 2,
            "eb_jurisdiction_id": [
                m.JURISDICTION_ID,
                m.JURISDICTION_ID,
                m.JURISDICTION_ID,
                "other",
            ],
        }
    ).to_parquet(bg, index=False)
    mask = m.freeze_support_mask(bg, tmp_path / "mask.csv")
    assert mask.tract_id.unique().tolist() == ["37001000100"]
    # Score preserves the fixed zero cell and returns finite exact metrics.
    truth = tmp_path / "truth.parquet"
    pd.DataFrame(
        {"tract_id": ["37001000100", "37001000300"], "incident_count": [3, 0]}
    ).to_parquet(truth, index=False)
    tract = tmp_path / "tract.parquet"
    pd.DataFrame(
        {
            "tract_id": ["37001000100", "37001000300"],
            "expected_count_rape": [2.0, 1.0],
            "primary_denominator_rape": [10.0, 5.0],
            "population_2025": [100.0, 50.0],
        }
    ).to_parquet(tract, index=False)
    arms = tmp_path / "arms.json"
    arms.write_text(
        json.dumps({"arms": [{"name": "candidate", "tract_path": str(tract)}]})
    )
    scores = m.score_truth(truth, arms)
    assert {r["surface"] for r in scores} == {
        "candidate",
        "primary_exposure__candidate",
        "uniform",
        "population",
    }
    assert all(r["support_cells"] == 2 and r["incidents"] == 3 for r in scores)


def test_report_date_epoch_uses_eastern_half_open_window():
    eastern = "America/New_York"
    instants = pd.Series(
        [
            pd.Timestamp("2026-01-01 00:00:00", tz=eastern).tz_convert("UTC").value // 10**6 - 1,
            pd.Timestamp("2026-01-01 00:00:00", tz=eastern).tz_convert("UTC").value // 10**6,
            pd.Timestamp("2026-07-01 00:00:00", tz=eastern).tz_convert("UTC").value // 10**6 - 1,
            pd.Timestamp("2026-07-01 00:00:00", tz=eastern).tz_convert("UTC").value // 10**6,
        ]
    )
    dates, mask = m.report_date_window_mask(instants)
    assert str(dates.dt.tz) == eastern
    assert mask.tolist() == [False, True, True, False]


def test_scoring_eligibility_has_no_unjustified_100_incident_cutoff():
    assert m.scoring_eligible(exact_point_violation=False, retained_incidents=1)
    assert m.scoring_eligible(exact_point_violation=False, retained_incidents=99)
    assert not m.scoring_eligible(exact_point_violation=False, retained_incidents=0)
    assert not m.scoring_eligible(exact_point_violation=True, retained_incidents=150)


def test_score_guard_binds_eligible_truth_manifest(tmp_path):
    truth = tmp_path / "truth.parquet"
    pd.DataFrame({"incident_count": [1]}).to_parquet(truth, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "truth_sha256": m.sha256(truth),
                "retention_audit": {"eligible_for_scoring": True},
            }
        )
    )
    m.require_scoring_eligibility(truth, manifest)
    manifest.write_text(
        json.dumps(
            {
                "truth_sha256": m.sha256(truth),
                "retention_audit": {"eligible_for_scoring": False},
            }
        )
    )
    with pytest.raises(ValueError, match="ineligible"):
        m.require_scoring_eligibility(truth, manifest)
