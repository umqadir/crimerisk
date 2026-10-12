"""No cache returns an adjudicated history or a holdout panel built from other inputs.

Two caches used to answer for the adjudicated history: a per-process memo keyed on the
repository, target year, row count and sum of preferred counts (so A=100/B=200 and
A=200/B=100 were the same panel), and the level holdout's preferred_panel_<year>.parquet,
reused on its file name alone. The memo is gone; the holdout cache is keyed on the full
identity of what it consumed. Every mutation below must miss.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import crimerisk.reporting_history as reporting_history  # noqa: E402
import crimerisk.trend_fills as trend_fills  # noqa: E402
from crimerisk.paths import RepoPaths  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "level_holdout_cache_under_test", REPO_ROOT / "scripts" / "eval" / "level_holdout.py"
)
holdout = importlib.util.module_from_spec(_SPEC)
sys.modules["level_holdout_cache_under_test"] = holdout
_SPEC.loader.exec_module(holdout)

YEAR = 2024
A, B = "GA0000001", "GA0000002"


def _observations(counts: dict[str, float], *, months: float = 12.0, usable: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ori9": list(counts),
            "year": YEAR,
            "source": "srs_return_a_annual",
            "offense": "burglary",
            "count": list(counts.values()),
            "months_reported": months,
            "usable": usable,
        }
    )


def _write_observations(paths: RepoPaths, frame: pd.DataFrame) -> None:
    path = paths.state_dir / "observations" / "agency_year_observations.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _repo(tmp_path: Path) -> RepoPaths:
    root = tmp_path / "repo"
    paths = RepoPaths.from_repo_root(root)
    configs = root / "configs" / "stage1_adjudications"
    configs.mkdir(parents=True)
    (root / "configs" / "level_lane_hard_evidence.csv").write_text("case_id,ori9,year\n")
    (configs / "zero_missing_adjudicated.csv").write_text("case_id,target_year,oris\n")
    source = root / "src" / "crimerisk"
    source.mkdir(parents=True)
    (source / "reporting_history.py").write_text("# v1\n")
    _write_observations(paths, _observations({A: 100.0, B: 200.0}))
    return paths


# --- the adjudicated history (build_agency_trend_fill_panel) has no memo -------------------


@pytest.fixture
def identity_history(monkeypatch):
    """The panel builder over stubbed preferred observations, with an identity adjudicator
    that records each call: what the builder returns is whatever it was last handed."""
    calls: list[tuple[str, pd.DataFrame]] = []
    current: dict[str, pd.DataFrame] = {}

    def preferred(*, paths, year, reporting_year_end, force_reporting_regimes_rebuild):
        frame = current["frame"]
        return frame[frame["year"].eq(year)].drop(columns="year").copy()

    def adjudicate(panel, *, paths, target_year):
        calls.append((str(paths.state_dir), panel.copy()))
        return panel.copy(), pd.DataFrame()

    monkeypatch.setattr(trend_fills, "build_agency_preferred_observations", preferred)
    monkeypatch.setattr(trend_fills, "add_preferred_support_flags", lambda frame: frame)
    monkeypatch.setattr(reporting_history, "adjudicate_history_years", adjudicate)

    def build(paths, rows: list[dict]) -> pd.DataFrame:
        current["frame"] = pd.DataFrame(rows)
        return trend_fills.build_agency_trend_fill_panel(paths=paths, year_start=YEAR, year_end=YEAR)

    return build, calls


def _rows(counts: dict[str, float], *, months: float = 12.0, usable: bool = True) -> list[dict]:
    return [
        {"ori9": ori, "year": YEAR, "state_fips": "13", "state_abbr": "GA", "offense": "burglary",
         "preferred_count": count, "preferred_months_reported": months, "usable_as_observed": usable}
        for ori, count in counts.items()
    ]


def _counts(panel: pd.DataFrame) -> dict[str, float]:
    return dict(zip(panel["ori9"].astype(str), panel["preferred_count"].astype(float)))


def test_history_same_sum_panels_are_adjudicated_separately(tmp_path, identity_history):
    build, calls = identity_history
    paths = RepoPaths.from_repo_root(tmp_path)
    first = build(paths, _rows({A: 100.0, B: 200.0}))
    second = build(paths, _rows({A: 200.0, B: 100.0}))
    assert _counts(first) == {A: 100.0, B: 200.0}
    assert _counts(second) == {A: 200.0, B: 100.0}
    assert len(calls) == 2


def test_history_months_only_change_is_adjudicated_again(tmp_path, identity_history):
    build, calls = identity_history
    paths = RepoPaths.from_repo_root(tmp_path)
    build(paths, _rows({A: 100.0, B: 200.0}))
    second = build(paths, _rows({A: 100.0, B: 200.0}, months=6.0))
    assert len(calls) == 2
    assert set(second["preferred_months_reported"]) == {6.0}


def test_history_usability_only_change_is_adjudicated_again(tmp_path, identity_history):
    build, calls = identity_history
    paths = RepoPaths.from_repo_root(tmp_path)
    build(paths, _rows({A: 100.0, B: 200.0}))
    second = build(paths, _rows({A: 100.0, B: 200.0}, usable=False))
    assert len(calls) == 2
    assert not second["usable_as_observed"].any()


def test_history_rule_only_and_fold_path_changes_are_adjudicated_again(tmp_path, identity_history):
    build, calls = identity_history
    paths = RepoPaths.from_repo_root(tmp_path / "edition")
    build(paths, _rows({A: 100.0, B: 200.0}))
    # Same panel, a changed dependency: the adjudicator runs again and reads it.
    build(paths, _rows({A: 100.0, B: 200.0}))
    # Same panel, a fold's state directory.
    fold = RepoPaths.from_repo_root(tmp_path / "fold")
    build(fold, _rows({A: 100.0, B: 200.0}))
    assert len(calls) == 3
    assert calls[-1][0] == str(fold.state_dir)
    assert not hasattr(reporting_history, "adjudicated_history_memo")


# --- the level holdout's preferred-panel cache is keyed on everything it consumed ----------


@pytest.fixture
def panel_builds(monkeypatch):
    """`build_preferred_panel` replaced by one that reads the observations it is keyed on."""
    calls: list[str] = []

    def build(paths, *, year, panel_year_end):
        calls.append(str(paths.state_dir))
        frame = pd.read_parquet(paths.state_dir / "observations" / "agency_year_observations.parquet")
        return frame, pd.DataFrame()

    monkeypatch.setattr(holdout, "build_preferred_panel", build)
    monkeypatch.delenv("CRIMERISK_LEVEL_RULES", raising=False)
    return calls


def _load(paths: RepoPaths, cache_dir: Path, **kwargs) -> pd.DataFrame:
    return holdout.load_preferred_panel(
        paths, year=YEAR, panel_year_end=2025, cache_dir=cache_dir, refresh=False, **kwargs
    )


def test_an_unchanged_input_set_reuses_the_cached_panel(tmp_path, panel_builds):
    paths = _repo(tmp_path)
    cache = tmp_path / "cache"
    _load(paths, cache)
    _load(paths, cache)
    assert len(panel_builds) == 1


def test_panel_cache_same_sum_swap_misses(tmp_path, panel_builds):
    paths = _repo(tmp_path)
    cache = tmp_path / "cache"
    _load(paths, cache)
    _write_observations(paths, _observations({A: 200.0, B: 100.0}))
    second = _load(paths, cache)
    assert len(panel_builds) == 2
    assert dict(zip(second["ori9"], second["count"])) == {A: 200.0, B: 100.0}


def test_panel_cache_months_only_change_misses(tmp_path, panel_builds):
    paths = _repo(tmp_path)
    cache = tmp_path / "cache"
    _load(paths, cache)
    _write_observations(paths, _observations({A: 100.0, B: 200.0}, months=6.0))
    second = _load(paths, cache)
    assert len(panel_builds) == 2
    assert set(second["months_reported"]) == {6.0}


def test_panel_cache_usability_only_change_misses(tmp_path, panel_builds):
    paths = _repo(tmp_path)
    cache = tmp_path / "cache"
    _load(paths, cache)
    _write_observations(paths, _observations({A: 100.0, B: 200.0}, usable=False))
    _load(paths, cache)
    # A usability directive (stage-1 ruling) changes with the data untouched.
    (paths.repo_root / "configs" / "stage1_adjudications" / "zero_missing_adjudicated.csv").write_text(
        f"case_id,target_year,oris\nz,{YEAR},{A}\n"
    )
    _load(paths, cache)
    assert len(panel_builds) == 3


def test_panel_cache_rule_only_change_misses(tmp_path, panel_builds, monkeypatch):
    paths = _repo(tmp_path)
    cache = tmp_path / "cache"
    _load(paths, cache)
    (paths.repo_root / "configs" / "level_lane_hard_evidence.csv").write_text(
        f"case_id,ori9,year\nh,{A},{YEAR}\n"
    )
    _load(paths, cache)
    monkeypatch.setenv("CRIMERISK_LEVEL_RULES", "none")
    _load(paths, cache)
    (paths.repo_root / "src" / "crimerisk" / "reporting_history.py").write_text("# v2\n")
    _load(paths, cache)
    assert len(panel_builds) == 4


def test_panel_cache_fold_path_change_misses(tmp_path, panel_builds):
    paths = _repo(tmp_path)
    cache = tmp_path / "cache"
    _load(paths, cache)
    # Identical content under another state directory is another fold's input.
    other_root = tmp_path / "other"
    shutil.copytree(paths.repo_root, other_root)
    _load(RepoPaths.from_repo_root(other_root), cache)
    _load(paths, cache, fold_id="fold_3")
    assert panel_builds == [str(paths.state_dir), str(RepoPaths.from_repo_root(other_root).state_dir), str(paths.state_dir)]


def test_panel_cache_year_and_tampering_miss(tmp_path, panel_builds):
    paths = _repo(tmp_path)
    cache = tmp_path / "cache"
    _load(paths, cache)
    holdout.load_preferred_panel(paths, year=YEAR, panel_year_end=2026, cache_dir=cache, refresh=False)
    assert len(panel_builds) == 2
    for stored in cache.rglob("preferred_panel_*.parquet"):
        _observations({A: 1.0}).to_parquet(stored, index=False)
    _load(paths, cache)
    assert len(panel_builds) == 3


def test_allocation_process_memos_are_keyed_on_the_state_read_and_the_rules(tmp_path, monkeypatch):
    import dataclasses

    import crimerisk.allocation as allocation

    production = RepoPaths.from_repo_root(tmp_path)
    fold = dataclasses.replace(production, state_dir=tmp_path / "fold" / "state", cache_dir=tmp_path / "fold" / "state" / "cache")
    monkeypatch.delenv("CRIMERISK_LEVEL_RULES", raising=False)
    base = allocation._agency_memo_key(production, YEAR)
    assert allocation._agency_memo_key(production, YEAR) == base
    assert allocation._agency_memo_key(fold, YEAR) != base
    assert allocation._agency_memo_key(production, YEAR - 1) != base
    monkeypatch.setenv("CRIMERISK_LEVEL_RULES", "none")
    assert allocation._agency_memo_key(production, YEAR) != base

    # The risk-signal memo answers only for the information set it was built from.
    allocation._AGENCY_RISK_SIGNAL_MEMO.clear()
    monkeypatch.delenv("CRIMERISK_LEVEL_RULES", raising=False)
    allocation._AGENCY_RISK_SIGNAL_MEMO[base] = pd.DataFrame({"production": [1]})
    calls = []
    monkeypatch.setattr(allocation, "build_agency_trend_fill_panel", lambda **kw: calls.append(kw) or (_ for _ in ()).throw(RuntimeError("rebuilt")))
    with pytest.raises(RuntimeError, match="rebuilt"):
        allocation._build_agency_risk_signals(paths=fold, year=YEAR)
    assert calls and allocation._build_agency_risk_signals(paths=production, year=YEAR).columns.tolist() == ["production"]
    allocation._AGENCY_RISK_SIGNAL_MEMO.clear()
