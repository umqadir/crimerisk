from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnostics" / "forward_chicago_2026.py"
spec = importlib.util.spec_from_file_location("forward_chicago_2026", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_query_is_bounded_start_inclusive_end_exclusive():
    params = module.socrata_params(limit=123, offset=456)
    assert params["$where"] == (
        "date >= '2026-01-01T00:00:00' AND date < '2026-07-01T00:00:00'"
    )
    assert params["$select"].split(",") == list(module.CHICAGO_RESOURCE_COLUMN_MAP)
    assert params["$order"] == "id"
    assert params["$limit"] == "123"
    assert params["$offset"] == "456"


def test_gate_rejects_unfrozen_and_post_2025_fit(tmp_path: Path):
    candidate = tmp_path / "manifest.json"
    candidate.write_text('{"year": 2025}')
    audit = tmp_path / "fit_cutoff_audit.json"
    audit.write_text('{"max_fitted_or_selected_crime_event_year": 2025}')
    base = {
        "candidate_frozen": True, "evaluation_contract_frozen": True,
        "target_window": "2026-01-01/2026-06-30",
        "evaluation_type": "conditional_spatial_distribution",
        "fit_effective_max_year": 2025,
        "eligible_offenses": ["murder", "robbery"],
        "target_event_holdout": True,
        "prospective_as_of_2025_12_31": False,
        "claim_label": "retrospective out-of-period target-event spatial validation",
        "fit_cutoff_audit_path": str(audit),
        "fit_cutoff_audit_sha256": module.sha256(audit),
        "candidate_manifest_sha256": module.sha256(candidate),
    }
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({**base, "candidate_frozen": False}))
    with pytest.raises(ValueError, match="true gates"):
        module.validate_gate(contract, candidate)
    contract.write_text(json.dumps({**base, "fit_effective_max_year": 2026}))
    with pytest.raises(ValueError, match="effective fit cutoff"):
        module.validate_gate(contract, candidate)
    contract.write_text(json.dumps({**base, "prospective_as_of_2025_12_31": True}))
    with pytest.raises(ValueError, match="must not claim prospective"):
        module.validate_gate(contract, candidate)
    contract.write_text(json.dumps({**base, "fit_cutoff_audit_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="Fit-cutoff audit"):
        module.validate_gate(contract, candidate)
    contract.write_text(json.dumps(base))
    assert module.validate_gate(contract, candidate)["candidate_frozen"] is True


def test_half_open_window_excludes_july_and_deduplicates(tmp_path: Path):
    page = tmp_path / "page.csv"
    pd.DataFrame([
        {"id":"1","date":"2026-01-01T00:00:00"},
        {"id":"1","date":"2026-01-01T00:00:00"},
        {"id":"2","date":"2026-07-01T00:00:00"},
    ]).to_csv(page,index=False)
    raw = pd.read_csv(page,dtype=str)
    raw["incident_dt"] = pd.to_datetime(raw["date"],utc=True)
    keep = raw.incident_dt.ge(pd.Timestamp(module.START,tz="UTC")) & raw.incident_dt.lt(
        pd.Timestamp(module.END_EXCLUSIVE,tz="UTC")
    )
    assert raw[keep].drop_duplicates("id")["id"].tolist() == ["1"]


def test_acquisition_pages_and_hashes_mocked_responses(tmp_path: Path, monkeypatch):
    payloads = [
        b"id,date\n1,2026-01-01T00:00:00\n2,2026-02-01T00:00:00\n",
        b"id,date\n3,2026-03-01T00:00:00\n",
    ]
    urls: list[str] = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def fake_urlopen(request, timeout):
        urls.append(request.full_url)
        return Response(payloads[len(urls) - 1])

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    records = module.acquire_pages(tmp_path, page_size=2)
    assert [record["rows"] for record in records] == [2, 1]
    assert [record["offset"] for record in records] == [0, 2]
    assert all(len(str(record["sha256"])) == 64 for record in records)
    assert "%24offset=0" in urls[0] and "%24offset=2" in urls[1]
