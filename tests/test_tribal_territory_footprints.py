"""Territory-basis custom footprints (`territory_prior`).

A tribal footprint covers every block group of the reservation / trust land, populated or not.
Each row carries per-leg territory shares; the agency's target is split by BG offense prior x the
offense's denominator-weighted territory share. The fixture is the Poarch Creek shape: one
residential reservation block group and one zero-resident casino block group whose destination
POIs and daytime population sit partly on trust land.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import crimerisk.allocation as allocation
from crimerisk.allocation import (
    CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT,
    CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_TERRITORY_PRIOR,
    _build_exclusive_footprint_displacement,
    _build_overlap_allocations,
    _custom_footprint_component_shares,
    _load_overlap_custom_footprints,
)
from crimerisk.exposure_ensemble import load_ensemble_weights
from crimerisk.paths import RepoPaths, get_paths
from crimerisk.territory_shares import (
    TERRITORY_LEGS,
    TERRITORY_OFFENSE_SHARE_COLUMN,
    TERRITORY_SHARE_COLUMNS,
    attach_territory_offense_shares,
    territory_leg_levels,
)

ORI = "AL0000000"
RESIDENTIAL = "010010001001"
CASINO = "010010001002"
OFFENSES = ("larceny", "burglary", "murder")
PRIOR = {
    ("larceny", RESIDENTIAL): 5.0,
    ("larceny", CASINO): 12.0,
    ("burglary", RESIDENTIAL): 1.0,
    ("burglary", CASINO): 2.0,
    # No model signal anywhere: the pool splits by territory shares.
    ("murder", RESIDENTIAL): 0.0,
    ("murder", CASINO): 0.0,
}
TARGETS = {"larceny": 443.0, "burglary": 20.0, "murder": 1.0}
# Per-leg shares: the casino part holds a fifth of its BG's POIs, a seventh of its daytime
# population and no residents; the residential BG has no POIs or retail jobs at all.
SHARES = {
    RESIDENTIAL: {"residents": 0.276, "landscan_day": 0.38, "daytime_jobs": 0.276,
                  "destination_poi": np.nan, "retail_jobs": np.nan, "industrial_jobs": 0.0,
                  "wac_jobs": 0.62, "housing_units": 0.26, "land": 0.0085},
    CASINO: {"residents": 0.0, "landscan_day": 0.13, "daytime_jobs": 0.0,
             "destination_poi": 0.2, "retail_jobs": 0.4, "industrial_jobs": 0.0,
             "wac_jobs": 0.006, "housing_units": 0.0, "land": 0.0012},
}


def _covariates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bg_id": [RESIDENTIAL, CASINO],
            "population": [1000.0, 700.0],
            "landscan_day_pop": [570.0, 2900.0],
            "daytime_population_jobs_proxy": [1000.0, 2100.0],
            "destination_poi_total": [0.0, 20.0],
            "lodes_retail_jobs": [0.0, 30.0],
            "lodes_industrial_jobs": [20.0, 100.0],
            "households_total": [400.0, 300.0],
            "burglary_destination_poi_exposure_weight": [24.0, 24.0],
            "burglary_retail_jobs_exposure_weight": [0.3, 0.3],
            "burglary_industrial_jobs_exposure_weight": [0.04, 0.04],
            "burglary_group_quarters_exposure_weight": [0.27, 0.27],
            "census2020_gq_total": [0.0, 0.0],
            "aggregate_vehicles_total": [700.0, 500.0],
            "mvt_commuter_vehicle_proxy": [60.0, 1800.0],
            "vehicle_exposure_2024": [760.0, 2300.0],
        }
    )


def _levels() -> pd.DataFrame:
    weights = load_ensemble_weights(get_paths().repo_root / "configs" / "exposure_ensemble_weights_v1.csv")
    return territory_leg_levels(_covariates(), weights=weights)


def _row(bg: str, basis: str, weight: float, *, responsibility: float | None = None) -> dict:
    row = {
        "state_fips": "01",
        "bg_id": bg,
        "ori9": ORI,
        "weight_share": weight,
        "weight_share_basis": basis,
        "bg_responsibility_population_coverage_share": responsibility,
        "responsibility_fraction_basis": "test_block_population" if responsibility else pd.NA,
        "allocation_scope": "source_state",
        "service_scope_id": pd.NA,
        "canonical_target_ori": ORI,
        "source_state_fips": "01",
    }
    for leg in (*TERRITORY_LEGS, "land"):
        row[f"territory_share_{leg}"] = SHARES[bg][leg] if basis.endswith("territory_prior") else np.nan
    return row


def _territory_footprint() -> pd.DataFrame:
    land = SHARES[RESIDENTIAL]["land"] + SHARES[CASINO]["land"]
    return pd.DataFrame(
        [
            _row(RESIDENTIAL, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_TERRITORY_PRIOR, SHARES[RESIDENTIAL]["land"] / land),
            _row(CASINO, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_TERRITORY_PRIOR, SHARES[CASINO]["land"] / land),
        ]
    )


def _allocate(monkeypatch, tmp_path, footprints: pd.DataFrame, levels=None) -> pd.DataFrame:
    targets = pd.DataFrame(
        [
            {
                "state_fips": "01",
                "offense": offense,
                "group_kind": "custom_footprint_overlap",
                "group_id": ORI,
                "target_count": TARGETS[offense],
            }
            for offense in OFFENSES
        ]
    )
    prior = pd.DataFrame(
        [
            {"state_fips": "01", "bg_id": bg, "tract_id": bg[:11], "offense": offense, "bg_weight": weight}
            for (offense, bg), weight in PRIOR.items()
        ]
    )
    monkeypatch.setattr(allocation, "_build_overlap_group_targets", lambda **_: targets)
    monkeypatch.setattr(allocation, "_load_overlap_custom_footprints", lambda _: footprints)
    monkeypatch.setattr(allocation, "_load_overlap_footprint_overrides", lambda _: pd.DataFrame())
    monkeypatch.setattr(
        allocation, "_load_concurrent_jurisdiction_carveouts", lambda _: pd.DataFrame()
    )
    return _build_overlap_allocations(
        paths=RepoPaths.from_repo_root(tmp_path),
        bg_prior=prior,
        bg_crosswalk=pd.DataFrame(),
        controls=pd.DataFrame(),
        year=2025,
        territory_leg_levels=levels,
    )


def _offense_share(bg: str, offense: str) -> float:
    rows = _territory_footprint().assign(offense=offense)
    out = attach_territory_offense_shares(rows, _levels())
    return float(out.loc[out["bg_id"].eq(bg), TERRITORY_OFFENSE_SHARE_COLUMN].iloc[0])


def test_larceny_share_weights_legs_by_the_bg_denominator():
    levels = _levels().set_index(["bg_id", "offense", "leg"])["level"]
    legs = ["residents", "landscan_day", "daytime_jobs", "destination_poi", "retail_jobs"]
    a = {leg: levels[(CASINO, "larceny", leg)] for leg in legs}
    expected = sum(a[leg] * SHARES[CASINO][leg] for leg in legs) / sum(a.values())
    assert _offense_share(CASINO, "larceny") == pytest.approx(expected)
    # Person ensemble 0.5 x (0.4, 0.3, 0.3): residents 0.2 x 700.
    assert a["residents"] == pytest.approx(0.5 * 0.4 * 700.0)
    # The POI leg is this BG's largest larceny leg, so its share leads the combination.
    assert a["destination_poi"] == max(a.values())


def test_a_leg_the_block_group_lacks_gets_no_weight():
    # The residential BG has no POIs or retail jobs: its larceny share is its person share.
    levels = _levels().set_index(["bg_id", "offense", "leg"])["level"]
    person = ["residents", "landscan_day", "daytime_jobs"]
    a = {leg: levels[(RESIDENTIAL, "larceny", leg)] for leg in person}
    expected = sum(a[leg] * SHARES[RESIDENTIAL][leg] for leg in person) / sum(a.values())
    assert levels[(RESIDENTIAL, "larceny", "destination_poi")] == 0.0
    assert _offense_share(RESIDENTIAL, "larceny") == pytest.approx(expected)


def test_offense_shares_differ_by_offense():
    # Murder's denominator is 0.9 residents + 0.1 daytime jobs: no casino share at all.
    assert _offense_share(CASINO, "murder") == pytest.approx(0.0)
    assert _offense_share(CASINO, "larceny") > 0.1
    assert _offense_share(CASINO, "motor_vehicle_theft") > 0.0


def test_land_share_applies_only_where_the_bg_has_no_leg():
    rows = _territory_footprint().assign(offense="larceny")
    levels = _levels()
    levels.loc[levels["bg_id"].eq(CASINO), "level"] = 0.0
    out = attach_territory_offense_shares(rows, levels).set_index("bg_id")
    assert out.loc[CASINO, TERRITORY_OFFENSE_SHARE_COLUMN] == pytest.approx(SHARES[CASINO]["land"])
    assert out.loc[CASINO, "territory_offense_share_basis"] == "land"


def test_zero_resident_activity_bg_inside_the_territory_receives_mass(monkeypatch, tmp_path):
    before = _allocate(
        monkeypatch,
        tmp_path,
        pd.DataFrame(
            [_row(RESIDENTIAL, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT, 1.0, responsibility=0.276)]
        ),
    )
    after = _allocate(monkeypatch, tmp_path, _territory_footprint(), _levels())

    larceny_before = before[before["offense"].eq("larceny")].set_index("bg_id")["component_count"]
    larceny_after = after[after["offense"].eq("larceny")].set_index("bg_id")["component_count"]
    assert larceny_before.to_dict() == pytest.approx({RESIDENTIAL: 443.0})
    # Both block groups are partial: each blends its prior with the footprint's rate x exposure.
    s_res, s_cas = _offense_share(RESIDENTIAL, "larceny"), _offense_share(CASINO, "larceny")
    e_res, e_cas = _exposure(RESIDENTIAL, "larceny"), _exposure(CASINO, "larceny")
    rate = (5.0 * s_res + 12.0 * s_cas) / (e_res + e_cas)
    w_res = s_res * 5.0 * s_res + (1 - s_res) * rate * e_res
    w_cas = s_cas * 12.0 * s_cas + (1 - s_cas) * rate * e_cas
    assert larceny_after[CASINO] == pytest.approx(443.0 * w_cas / (w_res + w_cas))
    assert larceny_after[CASINO] > larceny_after[RESIDENTIAL]


def test_territory_footprint_conserves_each_offense_target(monkeypatch, tmp_path):
    out = _allocate(monkeypatch, tmp_path, _territory_footprint(), _levels())

    totals = out.groupby("offense")["component_count"].sum().to_dict()
    assert totals == pytest.approx(TARGETS, abs=1e-12)
    # Murder has no prior signal: its territory shares split it, and the casino has none.
    murder = out[out["offense"].eq("murder")].set_index("bg_id")["component_count"]
    assert murder[RESIDENTIAL] == pytest.approx(1.0)


def test_zero_signal_pool_with_no_territory_share_falls_back_to_land():
    frame = _territory_footprint().assign(offense="murder", bg_weight=0.0)
    frame[TERRITORY_OFFENSE_SHARE_COLUMN] = 0.0
    frame["territory_offense_exposure"] = 0.0
    out = _custom_footprint_component_shares(frame)
    assert out["component_share"].tolist() == pytest.approx(frame["weight_share"].tolist())


def test_wholly_residential_footprint_is_unchanged_in_total(monkeypatch, tmp_path):
    # Every footprint BG lies wholly on the territory: every leg share 1 = responsibility 1.
    resident = pd.DataFrame(
        [
            _row(RESIDENTIAL, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT, 0.4, responsibility=1.0),
            _row(CASINO, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT, 0.6, responsibility=1.0),
        ]
    )
    territory = _territory_footprint()
    for column in TERRITORY_SHARE_COLUMNS:
        territory[column] = 1.0
    before = _allocate(monkeypatch, tmp_path, resident)
    after = _allocate(monkeypatch, tmp_path, territory, _levels())

    keys = ["offense", "bg_id"]
    before = before[before["offense"].ne("murder")].sort_values(keys).reset_index(drop=True)
    after = after[after["offense"].ne("murder")].sort_values(keys).reset_index(drop=True)
    assert after["component_count"].tolist() == pytest.approx(before["component_count"].tolist())
    assert after.groupby("offense")["component_count"].sum().to_dict() == pytest.approx(
        {"larceny": 443.0, "burglary": 20.0}
    )


def test_territory_rows_need_leg_levels(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="territory_leg_levels"):
        _allocate(monkeypatch, tmp_path, _territory_footprint(), None)


def test_territory_basis_is_ordinary_only():
    frame = _territory_footprint().assign(offense="larceny", bg_weight=1.0)
    frame[TERRITORY_OFFENSE_SHARE_COLUMN] = 0.5
    frame["allocation_scope"] = "service_wide"
    with pytest.raises(ValueError, match="territory_prior is only valid for ordinary"):
        _custom_footprint_component_shares(frame)


def test_mvt_presence_floor_uses_the_person_presence_leg():
    covariates = _covariates()
    covariates.loc[1, "vehicle_exposure_2024"] = 5000.0  # floor binds in the casino BG
    weights = load_ensemble_weights(get_paths().repo_root / "configs" / "exposure_ensemble_weights_v1.csv")
    levels = territory_leg_levels(covariates, weights=weights).set_index(["bg_id", "offense", "leg"])["level"]
    assert levels[(CASINO, "motor_vehicle_theft", "landscan_day")] == pytest.approx(5000.0)
    assert levels[(CASINO, "motor_vehicle_theft", "housing_units")] == 0.0
    assert levels[(RESIDENTIAL, "motor_vehicle_theft", "housing_units")] == pytest.approx(700.0)


def _write_footprints(tmp_path, rows: list[dict]) -> RepoPaths:
    config = tmp_path / "configs"
    config.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(config / "overlap_custom_footprints.csv", index=False)
    # A scope registry for some other agency, so the loader attaches every coverage column.
    pd.DataFrame(
        [
            {
                "service_scope_id": "other_service",
                "canonical_target_ori": "ZZ0000000",
                "source_state_fips": "04",
                "geometry_contributor_ori": "ZZ0000000",
                "allocation_scope": "service_wide",
                "official_source_ref": "test",
                "evidence_artifact": "test",
                "evidence_sha256": "test",
                "scope_note": "test",
            }
        ]
    ).to_csv(config / "service_wide_agency_scopes.csv", index=False)
    return RepoPaths.from_repo_root(tmp_path)


def _config_row(bg: str, basis: str, weight: float, coverage: float, *, legs: bool = True) -> dict:
    row = {
        "ori": ORI,
        "state_fips": "01",
        "block_group_geoid": bg,
        "weight_share": weight,
        "bg_population_coverage_share": coverage,
        "weight_share_basis": basis,
        "geometry_source_type": "test",
        "geometry_source_ref": "test",
        "footprint_note": "test",
    }
    if legs:
        for leg in (*TERRITORY_LEGS, "land"):
            row[f"territory_share_{leg}"] = SHARES[bg][leg]
    return row


def test_loader_accepts_zero_resident_territory_rows_and_they_displace_nothing(tmp_path):
    paths = _write_footprints(
        tmp_path,
        [
            _config_row(RESIDENTIAL, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_TERRITORY_PRIOR, 0.88, 0.276),
            _config_row(CASINO, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_TERRITORY_PRIOR, 0.12, 0.0),
        ],
    )
    footprints = _load_overlap_custom_footprints(paths)
    assert set(footprints["bg_id"]) == {RESIDENTIAL, CASINO}
    assert footprints.set_index("bg_id").loc[CASINO, "territory_share_destination_poi"] == 0.2

    displacement = _build_exclusive_footprint_displacement(
        overrides=pd.DataFrame({"ori9": [ORI], "displaces_county_remainder": [True]}),
        custom_footprints=footprints,
    )
    assert displacement.set_index("bg_id")["displaced_share"].to_dict() == pytest.approx(
        {RESIDENTIAL: 0.276}
    )


def test_loader_reads_a_file_without_territory_columns(tmp_path):
    paths = _write_footprints(
        tmp_path,
        [
            _config_row(RESIDENTIAL, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT, 1.0, 0.276, legs=False),
        ],
    )
    footprints = _load_overlap_custom_footprints(paths)
    assert footprints["territory_share_land"].isna().all()


def test_loader_requires_land_share_on_territory_rows(tmp_path):
    row = _config_row(RESIDENTIAL, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_TERRITORY_PRIOR, 1.0, 0.276)
    row["territory_share_land"] = np.nan
    paths = _write_footprints(tmp_path, [row])
    with pytest.raises(ValueError, match="territory_share_land"):
        _load_overlap_custom_footprints(paths)


def test_loader_still_rejects_zero_coverage_on_resident_rows(tmp_path):
    paths = _write_footprints(
        tmp_path,
        [
            _config_row(RESIDENTIAL, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT, 0.78, 0.276, legs=False),
            _config_row(CASINO, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT, 0.22, 0.0, legs=False),
        ],
    )
    with pytest.raises(ValueError, match="bg_population_coverage_share must lie"):
        _load_overlap_custom_footprints(paths)


# ------------------------------------- rule (c): whole BGs keep the prior, partial parts the rate

WHOLE = "010010001003"


def _exposure(bg: str, offense: str) -> float:
    rows = _territory_footprint().assign(offense=offense)
    out = attach_territory_offense_shares(rows, _levels())
    return float(out.loc[out["bg_id"].eq(bg), "territory_offense_exposure"].iloc[0])


def _pool(whole_prior: float, partial_prior: float, whole_exposure: float, partial_exposure: float,
          partial_share: float = 0.2) -> pd.DataFrame:
    rows = []
    for bg, prior, exposure, share, whole in (
        (WHOLE, whole_prior, whole_exposure, 1.0, True),
        (CASINO, partial_prior, partial_exposure, partial_share, False),
    ):
        row = _row(bg if bg != WHOLE else RESIDENTIAL, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_TERRITORY_PRIOR, 0.5)
        row["bg_id"] = bg
        for column in TERRITORY_SHARE_COLUMNS:
            row[column] = 1.0 if whole else partial_share
        row.update(offense="larceny", bg_weight=prior, territory_offense_share=share,
                   territory_offense_exposure=exposure)
        rows.append(row)
    return pd.DataFrame(rows)


def _blend(prior: float, share: float, exposure: float, rate: float) -> float:
    return share * prior * share + (1.0 - share) * rate * exposure


def test_weight_blends_prior_and_pool_rate_by_territory_share():
    frame = _pool(whole_prior=40.0, partial_prior=12.0, whole_exposure=1000.0, partial_exposure=500.0)
    out = _custom_footprint_component_shares(frame).set_index("bg_id")
    rate = (40.0 * 1.0 + 12.0 * 0.2) / 1500.0
    whole_w, partial_w = 40.0, _blend(12.0, 0.2, 500.0, rate)
    assert out.loc[WHOLE, "component_share"] == pytest.approx(whole_w / (whole_w + partial_w))
    assert out.loc[CASINO, "component_share"] == pytest.approx(partial_w / (whole_w + partial_w))
    assert out.loc[CASINO, "territory_part_rule"] == "prior_blend"
    assert out.loc[CASINO, "territory_part_prior_weight"] == pytest.approx(0.2)


def test_share_one_reproduces_prior_times_share_exactly():
    frame = _pool(40.0, 12.0, 1000.0, 500.0, partial_share=1.0)
    out = _custom_footprint_component_shares(frame)
    assert out["footprint_activity_weight"].tolist() == [40.0, 12.0]


def test_weight_is_continuous_in_the_territory_share():
    weights = []
    for share in (0.999999, 1.0):
        out = _custom_footprint_component_shares(_pool(40.0, 12.0, 1000.0, 500.0, partial_share=share))
        weights.append(out.set_index("bg_id").loc[CASINO, "footprint_activity_weight"])
    assert weights[0] == pytest.approx(weights[1], rel=1e-5)
    near = _custom_footprint_component_shares(_pool(40.0, 12.0, 1000.0, 500.0, partial_share=0.95))
    # 95% inside keeps nearly all of its own prior: 0.95 x 12 x 0.95 of a 0.95-share part.
    w = near.set_index("bg_id").loc[CASINO, "footprint_activity_weight"]
    assert w == pytest.approx(_blend(12.0, 0.95, 500.0, (40.0 + 12.0 * 0.95) / 1500.0))


def test_small_share_tends_to_the_pool_rate():
    frame = _pool(40.0, 12.0, 1000.0, 500.0, partial_share=1e-9)
    out = _custom_footprint_component_shares(frame).set_index("bg_id")
    rate = 40.0 / 1500.0
    assert out.loc[CASINO, "footprint_activity_weight"] == pytest.approx(rate * 500.0, rel=1e-6)


def test_partial_part_gets_mass_in_proportion_to_its_exposure():
    small = _custom_footprint_component_shares(_pool(40.0, 12.0, 1000.0, 250.0)).set_index("bg_id")
    large = _custom_footprint_component_shares(_pool(40.0, 12.0, 1000.0, 500.0)).set_index("bg_id")
    assert large.loc[CASINO, "component_share"] > small.loc[CASINO, "component_share"]


def test_rule_c_conserves_each_offense(monkeypatch, tmp_path):
    out = _allocate(monkeypatch, tmp_path, _territory_footprint(), _levels())
    assert out.groupby("offense")["component_count"].sum().to_dict() == pytest.approx(TARGETS, abs=1e-12)


def test_footprint_wholly_inside_is_prior_times_share():
    frame = _pool(40.0, 12.0, 1000.0, 500.0, partial_share=1.0)
    out = _custom_footprint_component_shares(frame)
    assert out["component_share"].tolist() == pytest.approx([40.0 / 52.0, 12.0 / 52.0])


def test_resident_and_service_bases_are_unchanged():
    resident = pd.DataFrame(
        [_row(RESIDENTIAL, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT, 0.6, responsibility=0.25),
         _row(CASINO, CUSTOM_FOOTPRINT_WEIGHT_SHARE_BASIS_RESIDENT, 0.4, responsibility=1.0)]
    ).assign(offense="larceny", bg_weight=[8.0, 2.0])
    out = _custom_footprint_component_shares(resident)
    assert out["component_share"].tolist() == pytest.approx([2.0 / 4.0, 2.0 / 4.0])
    assert "territory_part_rule" not in out.columns


def test_rule_c_pool_without_prior_signal_falls_back_to_territory_shares():
    frame = _pool(0.0, 0.0, 1000.0, 500.0)
    out = _custom_footprint_component_shares(frame).set_index("bg_id")
    assert out["component_share"].sum() == pytest.approx(1.0)
    assert out.loc[WHOLE, "component_share"] == pytest.approx(1.0 / 1.2)
