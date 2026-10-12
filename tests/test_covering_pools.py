import numpy as np
import pandas as pd

from crimerisk.covering_pools import RECIPIENT_COLUMN, spread_covering_pools, zero_total_towns
from crimerisk.crime import OFFENSES_7


def _controls(totals: dict[str, float]) -> pd.DataFrame:
    rows = []
    for jurisdiction_id, total in totals.items():
        for offense in OFFENSES_7:
            rows.append(
                {
                    "jurisdiction_id": jurisdiction_id,
                    "jurisdiction_type": "municipal"
                    if ":municipal:" in jurisdiction_id
                    else "state_nonmunicipal_remainder",
                    "offense": offense,
                    "adjusted_count_ags_core": total / len(OFFENSES_7),
                    "bucket_population": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _component(bg_id, jurisdiction_id, jurisdiction_type, offense, count, denominator, ori=pd.NA):
    return {
        "state_fips": bg_id[:2],
        "source_state_fips": bg_id[:2],
        "bg_id": bg_id,
        "tract_id": bg_id[:11],
        "jurisdiction_id": jurisdiction_id,
        "jurisdiction_type": jurisdiction_type,
        "offense": offense,
        "service_scope_id": pd.NA,
        "canonical_target_ori": ori,
        "source_target_count": pd.NA,
        "component_count": count,
        "primary_denominator_raw": denominator,
    }


def test_zero_total_town_takes_the_county_pool_rate_and_the_pool_keeps_its_total():
    town = "25:municipal:cousub:2501313485"
    remainder_bg, town_bg = "250138130012", "250138130011"
    rows = []
    for offense in OFFENSES_7:
        rows += [
            # the county-localized state-police subunit and the county remainder, both on
            # the one nonmunicipal block group
            _component(remainder_bg, "25013", "localized_overlap_county_nonmunicipal_layer", offense, 20.0, 1700.0),
            _component(remainder_bg, "25:state_nonmunicipal_remainder:county:25013", "localized_remainder_county_layer", offense, 1.0, 1700.0),
            _component(town_bg, town, "municipal", offense, 0.0, 1250.0),
            _component(town_bg, "25", "statewide_overlap_layer", offense, 0.01, 1250.0),
        ]
    components = pd.DataFrame(rows)
    # Stage 12's component table carries DataFrames in attrs (service_unlocated_mass).
    components.attrs["service_unlocated_mass"] = pd.DataFrame({"unlocated_count": [1.0, 2.0]})
    bg_crosswalk = pd.DataFrame(
        {
            "state_fips": ["25", "25"],
            "block_group_geoid": [remainder_bg, town_bg],
            "jurisdiction_id": ["25:state_nonmunicipal_remainder", town],
            "jurisdiction_type": ["state_nonmunicipal_remainder", "municipal"],
            "allocation_share": [1.0, 1.0],
        }
    )
    controls = _controls({town: 0.0, "25:municipal:cousub:2501370045": 50.0})
    assert list(zero_total_towns(controls)) == [town]
    out, audit = spread_covering_pools(
        components,
        controls=controls,
        bg_crosswalk=bg_crosswalk,
        agency_crosswalk=pd.DataFrame({"ori": ["MA0070400"], "jurisdiction_id": [town]}),
        agency_types=pd.Series(dtype="string"),
        covered_by=pd.DataFrame(columns=["ori9", "covering_ori"]),
    )
    larceny = out[out["offense"].eq("larceny")]
    per_bg = larceny.groupby("bg_id")["component_count"].sum()
    # pool 21 on 1,700 people; the town adds 1,250 at the pool rate -> equal rates after
    rate = 21.0 / (1700.0 + 1250.0)
    assert np.isclose(per_bg[remainder_bg], rate * 1700.0)
    assert np.isclose(per_bg[town_bg] - 0.01, rate * 1250.0)
    # every lineage keeps its total, and the recipients are labelled with the town
    for jurisdiction_id, before in components.groupby("jurisdiction_id")["component_count"].sum().items():
        assert np.isclose(out.loc[out["jurisdiction_id"].eq(jurisdiction_id), "component_count"].sum(), before)
    assert set(out[RECIPIENT_COLUMN].dropna()) == {town}
    assert set(audit["pool_kind"]) == {"county_nonmunicipal_pool"}
    assert isinstance(out.attrs["service_unlocated_mass"], pd.DataFrame)


def test_contract_town_takes_its_contracting_agency_before_the_county_pool():
    town, city = "27:municipal:place:2729258", "27:municipal:place:2712700"
    town_bg, city_bg, rem_bg = "270030513053", "270030513022", "270030501001"
    rows = []
    for offense in OFFENSES_7:
        rows += [
            _component(city_bg, city, "municipal", offense, 30.0, 3000.0),
            _component(town_bg, town, "municipal", offense, 0.0, 1000.0),
            _component(rem_bg, "27:state_nonmunicipal_remainder:county:27003", "localized_remainder_county_layer", offense, 5.0, 500.0),
        ]
    bg_crosswalk = pd.DataFrame(
        {
            "state_fips": ["27"] * 3,
            "block_group_geoid": [city_bg, town_bg, rem_bg],
            "jurisdiction_id": [city, town, "27:state_nonmunicipal_remainder"],
            "jurisdiction_type": ["municipal", "municipal", "state_nonmunicipal_remainder"],
            "allocation_share": [1.0, 1.0, 1.0],
        }
    )
    out, audit = spread_covering_pools(
        pd.DataFrame(rows),
        controls=_controls({town: 0.0, city: 210.0}),
        bg_crosswalk=bg_crosswalk,
        agency_crosswalk=pd.DataFrame({"ori": ["MN0021400", "MN0020400"], "jurisdiction_id": [town, city]}),
        agency_types=pd.Series(dtype="string"),
        covered_by=pd.DataFrame({"ori9": ["MN0021400"], "covering_ori": ["MN0020400"]}),
    )
    assert set(audit["pool_kind"]) == {"contracting_agency"}
    robbery = out[out["offense"].eq("robbery")].groupby("bg_id")["component_count"].sum()
    assert np.isclose(robbery[city_bg] + robbery[town_bg], 30.0)
    assert np.isclose(robbery[town_bg] / 1000.0, robbery[city_bg] / 3000.0)
    assert np.isclose(robbery[rem_bg], 5.0)


def test_a_donor_s_mass_off_its_own_ground_is_neither_pooled_nor_rescaled():
    """B2 leaves part of a state-police unit's mass on county block groups outside its footprint,
    under the unit's own id. The pool rate is the unit's mass over exposure on its own ground."""
    town = "42:municipal:cousub:4200300001"
    unit = "42:police_service:PAPSP0100"
    footprint_bg, town_bg, city_bg = "420030001001", "420030002001", "420030003001"
    city = "42:municipal:cousub:4200300002"
    rows = []
    for offense in OFFENSES_7:
        rows += [
            _component(footprint_bg, unit, "police_service", offense, 2.0, 500.0),
            # the footprint-coverage rule's county-wide share, on the city's block group
            _component(city_bg, unit, "police_service", offense, 8.0, 4000.0),
            _component(city_bg, city, "municipal", offense, 40.0, 4000.0),
            _component(town_bg, town, "municipal", offense, 0.0, 1500.0),
        ]
    bg_crosswalk = pd.DataFrame(
        {
            "state_fips": ["42"] * 3,
            "block_group_geoid": [footprint_bg, town_bg, city_bg],
            "jurisdiction_id": [unit, town, city],
            "jurisdiction_type": ["police_service", "municipal", "municipal"],
            "allocation_share": [1.0, 1.0, 1.0],
        }
    )
    out, audit = spread_covering_pools(
        pd.DataFrame(rows),
        controls=_controls({town: 0.0, city: 280.0}),
        bg_crosswalk=bg_crosswalk,
        agency_crosswalk=pd.DataFrame({"ori": ["PA0020100"], "jurisdiction_id": [town]}),
        agency_types=pd.Series({"PAPSP0100": "state_law_enforcement"}, dtype="string"),
        covered_by=pd.DataFrame(columns=["ori9", "covering_ori"]),
    )
    assert set(audit["pool_kind"]) == {"state_police_service"}
    burglary = out[out["offense"].eq("burglary") & out["jurisdiction_id"].eq(unit)]
    per_bg = burglary.groupby("bg_id")["component_count"].sum()
    rate = 2.0 / (500.0 + 1500.0)
    assert np.isclose(per_bg[footprint_bg], rate * 500.0)
    assert np.isclose(per_bg[town_bg], rate * 1500.0)
    assert np.isclose(per_bg[city_bg], 8.0)
    assert np.isclose(burglary["component_count"].sum(), 10.0)


def test_the_component_audit_follows_the_spread():
    """Review #1: confidence and provenance read the published (post-spread) allocation."""
    from crimerisk.allocation import COMPONENT_POSITION_COLUMN, _component_audit_after_covering_pools

    before = pd.DataFrame(
        [
            _component("250138130012", "25013", "localized_overlap_county_nonmunicipal_layer", "larceny", 20.0, 1700.0),
            _component("250138130011", "25:municipal:cousub:2501313485", "municipal", "larceny", 0.0, 1250.0),
        ]
    )
    before["source_target_count"] = [40.0, 0.0]
    audit = before.drop(columns=["component_count", "primary_denominator_raw"]).assign(
        component_count_before=[20.0, 0.0],
        component_count_after=[20.0, 0.0],
        component_share=[0.5, 0.0],
        model_share=[0.5, 0.0],
        city_incident_posterior_active=[False, False],
        audit_reason=["unchanged", "unchanged"],
        redistribution_status=["unchanged", "unchanged"],
    )
    audit[COMPONENT_POSITION_COLUMN] = [0, 1]
    audit = audit.iloc[::-1].reset_index(drop=True)  # the audit is sorted differently
    recipient = dict(before.iloc[0])
    recipient.update({"bg_id": "250138130011", "tract_id": "25013813001", "component_count": 8.0})
    after = pd.concat(
        [before.assign(component_count=[12.0, 0.0]), pd.DataFrame([recipient])], ignore_index=True
    )
    out = _component_audit_after_covering_pools(audit, before=before, after=after)
    assert COMPONENT_POSITION_COLUMN not in out.columns
    donor = out[out["bg_id"].eq("250138130012")].iloc[0]
    assert donor["component_count_after"] == 12.0 and np.isclose(donor["component_share"], 0.3)
    received = out[out["redistribution_status"].eq("covering_pool_recipient")]
    assert len(received) == 1
    row = received.iloc[0]
    # The recipient is a modeled component of its donor's lineage.
    assert row["jurisdiction_id"] == "25013" and row["component_count_after"] == 8.0
    assert np.isclose(row["component_share"], 0.2) and not bool(row["city_incident_posterior_active"])
    assert np.isclose(out["component_count_after"].sum(), 20.0)


def test_contract_relationships_come_from_the_latest_year_in_the_lookback(tmp_path):
    """Review #3: the batch-header archive ends before the release year."""
    from crimerisk.contract_coverage import covered_by_source_path, load_latest_covered_by_relationships
    from crimerisk.paths import RepoPaths

    paths = RepoPaths.from_repo_root(tmp_path)
    path = covered_by_source_path(paths)
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "ori": ["MA0070400", "MA0070400", "MA0070000"],
            "year": [2023, 2024, 2024],
            "population": [1000.0, 1000.0, 50000.0],
            "covered_by_ori": ["MA0070000", "MA0070000", ""],
        }
    ).to_parquet(path, index=False)
    frame = load_latest_covered_by_relationships(paths, year=2025)
    assert frame.attrs["relationship_year"] == 2024
    assert frame[["ori9", "covering_ori"]].values.tolist() == [["MA0070400", "MA0070000"]]
    assert load_latest_covered_by_relationships(paths, year=2031).attrs["relationship_year"] is None
