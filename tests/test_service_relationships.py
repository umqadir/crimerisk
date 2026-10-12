"""Who polices which territory comes from relationship rows, never from whether an ORI reports."""

from __future__ import annotations

import pandas as pd
import pytest

from crimerisk.covering_pools import covering_pool_recipients
from crimerisk.service_relationships import (
    DISPOSITION_MERGED,
    DISPOSITION_REMAINDER,
    SOURCE_CONFIG,
    SOURCE_FBI,
    ServiceRelationshipError,
    TABLE_COLUMNS,
    assert_territories_resolved,
    territory_check,
    territory_dispositions,
    territory_populations,
    validate_relationship_config,
    validate_relationship_territories,
)
from crimerisk.smoothed_controls import SmoothedControlConfig, build_smoothed_controls

SAN_ANSELMO = "06:municipal:place:0664434"
LARKSPUR = "06:municipal:place:0640438"
CORTE_MADERA = "06:municipal:place:0616462"
GREENVILLE = "28:municipal:place:2829180"
PATCHOGUE = "36:municipal:place:3656660"

CROSSWALK = pd.DataFrame(
    {
        "ori9": ["CA0211700", "CA0210600", "CA0210400", "MS0760100", "NY0511000", "NY0510100"],
        "jurisdiction_id": [SAN_ANSELMO, LARKSPUR, CORTE_MADERA, GREENVILLE, PATCHOGUE, "36:state_nonmunicipal_remainder"],
        "weight": [1.0] * 6,
    }
)


def _relationship(rid, territory, provider, relation="contract", source=SOURCE_CONFIG):
    return {
        "relationship_id": rid, "territory_jurisdiction_id": territory, "provider_ori9": provider,
        "relation": relation, "effective_from_year": "2013", "effective_to_year": pd.NA,
        "source": source, "evidence_source": "test",
    }


def test_a_multi_town_department_absorbs_its_member_towns():
    rels = pd.DataFrame(
        [
            _relationship("cm-larkspur", LARKSPUR, "CA0211700", "consolidated"),
            _relationship("cm-corte", CORTE_MADERA, "CA0211700", "consolidated"),
            # The provider's own town: nothing moves.
            _relationship("cm-san-anselmo", SAN_ANSELMO, "CA0211700", "consolidated"),
            # A county police department's contract town goes to the county pools.
            _relationship("patchogue", PATCHOGUE, "NY0510100", "contract", SOURCE_FBI),
        ],
        columns=TABLE_COLUMNS,
    )
    out = territory_dispositions(rels, crosswalk=CROSSWALK).set_index("territory_jurisdiction_id")
    assert out.loc[LARKSPUR, "disposition"] == DISPOSITION_MERGED
    assert out.loc[LARKSPUR, "principal_jurisdiction_id"] == SAN_ANSELMO
    assert out.loc[CORTE_MADERA, "principal_jurisdiction_id"] == SAN_ANSELMO
    assert SAN_ANSELMO not in out.index
    assert out.loc[PATCHOGUE, "disposition"] == DISPOSITION_REMAINDER
    # A non-reporting department with no relationship row keeps its own territory.
    assert GREENVILLE not in out.index


def test_two_providers_for_one_town_or_a_chained_move_is_refused():
    rels = pd.DataFrame(
        [_relationship("a", LARKSPUR, "CA0211700"), _relationship("b", LARKSPUR, "NY0510100")],
        columns=TABLE_COLUMNS,
    )
    with pytest.raises(ServiceRelationshipError, match="more than one"):
        territory_dispositions(rels, crosswalk=CROSSWALK)
    chained = pd.DataFrame(
        [_relationship("a", LARKSPUR, "CA0211700"), _relationship("b", SAN_ANSELMO, "MS0760100")],
        columns=TABLE_COLUMNS,
    )
    with pytest.raises(ServiceRelationshipError, match="itself moved"):
        territory_dispositions(chained, crosswalk=CROSSWALK)


def test_the_territory_check_fails_until_the_footprint_matches():
    population = pd.Series({SAN_ANSELMO: 12_830.0, LARKSPUR: 12_900.0, CORTE_MADERA: 10_000.0, GREENVILLE: 30_000.0})
    served = pd.Series({"CA0211700": 34_593.0, "MS0760100": 28_551.0})
    types = pd.Series({"CA0211700": "local_police", "MS0760100": "local_police"})
    empty = pd.DataFrame(columns=TABLE_COLUMNS)
    no_exceptions = pd.DataFrame(columns=["ori9"])
    flags = territory_check(
        served=served, crosswalk=CROSSWALK, jurisdiction_population=population,
        dispositions=territory_dispositions(empty, crosswalk=CROSSWALK), relationships=empty,
        exceptions=no_exceptions, agency_types=types,
    )
    assert flags.set_index("ori9").loc["CA0211700", "resolution"] == "unresolved"
    # served 28,551 vs 30,000: within 30%
    assert flags.set_index("ori9").loc["MS0760100", "resolution"] == "within_band"
    # An ORI the FBI lists as covered by another serves no territory of its own.
    covered = territory_check(
        served=served, crosswalk=CROSSWALK, jurisdiction_population=population,
        dispositions=territory_dispositions(empty, crosswalk=CROSSWALK), relationships=empty,
        exceptions=no_exceptions, agency_types=types, fbi_covered_oris=frozenset({"CA0211700"}),
    )
    assert "CA0211700" not in set(covered["ori9"])
    with pytest.raises(ServiceRelationshipError, match="no recorded"):
        assert_territories_resolved(flags)

    rels = pd.DataFrame(
        [_relationship("cm-larkspur", LARKSPUR, "CA0211700", "consolidated"),
         _relationship("cm-corte", CORTE_MADERA, "CA0211700", "consolidated")],
        columns=TABLE_COLUMNS,
    )
    # After the moves San Anselmo's geometry holds all three towns.
    after = pd.Series({SAN_ANSELMO: 35_730.0, GREENVILLE: 30_000.0})
    resolved = territory_check(
        served=served, crosswalk=CROSSWALK, jurisdiction_population=after,
        dispositions=territory_dispositions(rels, crosswalk=CROSSWALK), relationships=rels,
        exceptions=no_exceptions, agency_types=types,
    )
    # With its member towns mapped, Central Marin's territory matches what it serves.
    assert resolved.set_index("ori9").loc["CA0211700", "resolution"] == "within_band"
    assert_territories_resolved(resolved)


def _check(served, crosswalk, population, rels, exceptions=None, **kwargs):
    rels = pd.DataFrame(rels, columns=TABLE_COLUMNS)
    exceptions = exceptions if exceptions is not None else pd.DataFrame(columns=["ori9", "reason_code"])
    return territory_check(
        served=pd.Series(served), crosswalk=crosswalk, jurisdiction_population=pd.Series(population),
        dispositions=territory_dispositions(rels, crosswalk=crosswalk), relationships=rels,
        exceptions=exceptions, agency_types=pd.Series({o: "local_police" for o in served}), **kwargs,
    ).set_index("ori9")


def test_a_relationship_row_does_not_waive_a_residual_mismatch():
    town = "13:municipal:place:1300001"
    crosswalk = pd.DataFrame({"ori9": ["GA0000100"], "jurisdiction_id": [town], "weight": [1.0]})
    rels = [_relationship("own", town, "GA0000100", "exclusive")]
    flags = _check({"GA0000100": 100_000.0}, crosswalk, {town: 1_000.0}, rels)
    assert flags.loc["GA0000100", "resolution"] == "unresolved"
    assert flags.loc["GA0000100", "ratio"] == pytest.approx(100.0)
    with pytest.raises(ServiceRelationshipError, match="population-comparability exception"):
        assert_territories_resolved(flags.reset_index())
    excepted = _check(
        {"GA0000100": 100_000.0}, crosswalk, {town: 1_000.0}, rels,
        exceptions=pd.DataFrame({"ori9": ["GA0000100"], "reason_code": ["fbi_population_noise"]}),
    )
    assert excepted.loc["GA0000100", "resolution"] == "recorded_exception"
    assert_territories_resolved(excepted.reset_index())


def test_a_county_unincorporated_provider_is_checked_on_city_plus_county_remainder():
    city, county = "21:municipal:place:2179482", "21:state_nonmunicipal_remainder:county:21239"
    crosswalk = pd.DataFrame({"ori9": ["KY1200100"], "jurisdiction_id": [city], "weight": [1.0]})
    population = {city: 12_200.0, county: 15_000.0}
    rels = [_relationship("woodford", county, "KY1200100", "county_unincorporated")]
    flags = _check({"KY1200100": 27_242.0}, crosswalk, population, rels)
    assert flags.loc["KY1200100", "footprint_population"] == pytest.approx(27_200.0)
    assert flags.loc["KY1200100", "resolution"] == "within_band"
    # Without the county row the city alone is 2.2x short.
    assert _check({"KY1200100": 27_242.0}, crosswalk, population, []).loc["KY1200100", "resolution"] == "unresolved"
    # A custom footprint counts toward the footprint as well.
    custom = _check({"KY1200100": 27_242.0}, crosswalk, {city: 12_200.0}, [], custom_population=pd.Series({"KY1200100": 15_000.0}))
    assert custom.loc["KY1200100", "resolution"] == "within_band"


def test_a_partly_transferred_agency_stays_checked_on_what_it_keeps():
    own, contracted, provider_town = (f"27:municipal:place:27000{i}" for i in (1, 2, 3))
    crosswalk = pd.DataFrame(
        {"ori9": ["MN0000100", "MN0000100", "MN0000200", "MN0000300"],
         "jurisdiction_id": [own, contracted, contracted, provider_town], "weight": [1.0, 1.0, 1.0, 1.0]}
    )
    rels = [_relationship("c", contracted, "MN0000300")]
    # After the move the contracted town's population sits in the provider's jurisdiction.
    population = {own: 5_000.0, provider_town: 20_000.0}
    flags = _check({"MN0000100": 15_000.0, "MN0000200": 12_000.0, "MN0000300": 20_000.0}, crosswalk, population, rels)
    assert flags.loc["MN0000100", "footprint_population"] == pytest.approx(5_000.0)
    assert flags.loc["MN0000100", "resolution"] == "unresolved"
    # Every territory of MN0000200 is moved to another provider: it serves none of its own.
    assert "MN0000200" not in flags.index
    assert flags.loc["MN0000300", "resolution"] == "within_band"


def _config(*rows):
    columns = ["relationship_id", "territory_jurisdiction_id", "provider_ori9", "relation", "effective_from_year", "effective_to_year"]
    return pd.DataFrame([dict(zip(columns, row)) for row in rows], columns=columns).astype("string")


def test_relationship_rows_need_a_known_provider_strict_years_and_no_overlap():
    known = {"GA0250300", "GA0250100"}
    ok = _config(
        ("merged", "13:municipal:place:1369000", "GA0250100", "consolidated", "2005", "2017"),
        ("split", "13:municipal:place:1369000", "GA0250300", "exclusive", "2018", pd.NA),
    )
    validate_relationship_config(ok, known_oris=known)
    with pytest.raises(ServiceRelationshipError, match="absent from the agency master"):
        validate_relationship_config(_config(("x", LARKSPUR, "CA9999999", "contract", "2013", pd.NA)), known_oris=known)
    with pytest.raises(ServiceRelationshipError, match="ends before it starts"):
        validate_relationship_config(_config(("x", LARKSPUR, "GA0250300", "contract", "2020", "2019")), known_oris=known)
    with pytest.raises(ServiceRelationshipError, match="non-integer"):
        validate_relationship_config(_config(("x", LARKSPUR, "GA0250300", "contract", "2020.5", pd.NA)), known_oris=known)
    overlap = _config(
        ("merged", "13:municipal:place:1369000", "GA0250100", "consolidated", "2005", "2018"),
        ("split", "13:municipal:place:1369000", "GA0250300", "exclusive", "2018", pd.NA),
    )
    with pytest.raises(ServiceRelationshipError, match="overlapping years"):
        validate_relationship_config(overlap, known_oris=known)
    open_ended = _config(
        ("a", LARKSPUR, "GA0250100", "contract", pd.NA, pd.NA),
        ("b", LARKSPUR, "GA0250300", "contract", "2024", pd.NA),
    )
    with pytest.raises(ServiceRelationshipError, match="overlapping years"):
        validate_relationship_config(open_ended, known_oris=known)


def test_a_relationship_territory_must_resolve_to_population():
    bg = pd.DataFrame(
        {"block_group_geoid": ["212399999001", "212399999002", "211110001001"],
         "jurisdiction_id": ["21:state_nonmunicipal_remainder", "21:municipal:place:2179482", "21:state_nonmunicipal_remainder"],
         "pop20": [1_500.0, 2_000.0, 700.0]}
    )
    population = territory_populations(bg, column="pop20")
    assert population["21:state_nonmunicipal_remainder:county:21239"] == 1_500.0
    assert population["21:municipal:place:2179482"] == 2_000.0
    config = _config(("w", "21:state_nonmunicipal_remainder:county:21239", "KY1200100", "county_unincorporated", "2004", pd.NA))
    validate_relationship_territories(config, territory_population=population)
    for territory in ("21:state_nonmunicipal_remainder:county:21001", "21:municipal:place:2199999"):
        with pytest.raises(ServiceRelationshipError, match="no population"):
            validate_relationship_territories(
                _config(("w", territory, "KY1200100", "county_unincorporated", "2004", pd.NA)),
                territory_population=population,
            )


def _accounting(jid, count, mode):
    return {
        "jurisdiction_id": jid, "jurisdiction_type": "municipal", "jurisdiction_name": jid,
        "state_fips": "36", "state_abbr": "NY", "geo_type": "place", "geoid": jid[-7:], "offense": "larceny",
        "bucket_population": 21_000.0, "pop_band": "10k-25k", "adjusted_count_ags_core": count,
        "level_repair_mode": mode,
    }


def test_a_covered_town_is_not_revived_by_its_own_history_and_stays_a_pool_recipient():
    grand_island = "36:municipal:cousub:3602929828"
    peer = "36:municipal:cousub:3600000001"
    accounting = pd.DataFrame(
        [_accounting(grand_island, 0.0, "covered_by_covering_agency"), _accounting(peer, 300.0, "none")]
    )
    panel = pd.DataFrame(
        [
            {"jurisdiction_id": jid, "offense": "larceny", "year": year, "estimated_count": value,
             "full_year_admitted_count": value, "full_year_admitted_share": 1.0, "bucket_population": 21_000.0}
            for jid, value in ((grand_island, 220.0), (peer, 300.0))
            for year in range(2019, 2025)
        ]
    )
    out = build_smoothed_controls(accounting=accounting, panel=panel, config=SmoothedControlConfig(year=2025)).set_index("jurisdiction_id")
    assert out.loc[grand_island, "precalibration_smoothed_count"] == 0.0
    assert out.loc[grand_island, "fallback_reason"] == "covered_by_provider_protected"

    controls = accounting.assign(adjusted_count_ags_core=out["smoothed_count"].reindex(accounting["jurisdiction_id"]).to_numpy())
    assert grand_island in set(covering_pool_recipients(controls))
    assert peer not in set(covering_pool_recipients(controls))
    revived = controls.assign(adjusted_count_ags_core=[220.0, 300.0])
    with pytest.raises(ValueError, match="still carry a numerator"):
        covering_pool_recipients(revived)
