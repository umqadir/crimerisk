from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest
from openpyxl import Workbook

from scripts.pull.parse_fdle_fibrs_offense_detail import (
    MONTH_NAMES,
    parse_fdle_fibrs_offense_detail,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_0826 = REPO_ROOT / "data" / "FDLE-FIBRS-2024" / "raw" / "FIBRS_Offense_Detail_08-2026.xlsx"
OFFENSES = [
    "Murder and Non-Negligent Manslaughter",
    "Forcible Sex Offenses",
    "Robbery",
    "Aggravated Assault",
    "Burglary",
    "Larceny - Pocket Picking",
    "Larceny - Purse Snatching",
    "Larceny - Shoplifting",
    "Larceny - From Motor Vehicle",
    "Larceny - Motor Vehicle Parts",
    "Larceny - From Building",
    "Larceny - Coin-Operated Machine",
    "Larceny - All Other",
    "Motor Vehicle Theft",
    "Negligent Manslaughter",
    "Arson",
    "Simple Assault",
    "Stalking",
    "Intimidation",
    "Kidnapping/Abduction",
    "Drug Offenses",
    "Bribery",
    "Embezzlement",
    "Fraud",
    "Counterfeiting/Forgery",
    "Extortion/Blackmail",
    "Prostitutition",
    "Non-Forcible Sex Offenses",
    "Stolen Property Offenses",
    "Destruction/Damage/Vandalism",
    "Gambling Offenses",
    "Weapons Offenses",
    "Identity Theft",
    "Hacking/Computer Invasion",
    "Human Trafficking - Commercial Sex Acts",
    "Human Trafficking - Involuntary Servitude",
    "Animal Cruelty",
]


def _synthetic_workbook(path: Path, year: int) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = f"FIBRS Offense {year}"
    ws.append([f"{year} FIBRS Reported Offenses by Offense Type and Month"])
    offense_row = [None, None, f" {year} Population Estimate"]
    month_row = ["County", "Agency Name", None]
    for name in OFFENSES:
        offense_row += [name] + [None] * 11
        month_row += [f"{year} {m}" for m in MONTH_NAMES]
    ws.append(offense_row)
    ws.append(month_row)
    murder = [1, 0, None, None, None, None, None, None, None, None, None, 2]
    ws.append(["Alpha", "Alpha Police Dept.", 1000] + murder + [None] * 12 * (len(OFFENSES) - 1))
    ws.append([None, None, None])
    wb.save(path)


def test_parser_semantics_on_synthetic_sheet(tmp_path) -> None:
    path = tmp_path / "fibrs.xlsx"
    _synthetic_workbook(path, 2025)
    rows = parse_fdle_fibrs_offense_detail(path, 2025)
    # One agency (the blank trailing row is skipped) x 37 offenses x 13 periods.
    assert len(rows) == 37 * 13
    murder = {r["period"]: r for r in rows if r["offense_slug"] == "murder_and_non_negligent_manslaughter"}
    assert murder["YTD"]["incident_count"] == 3
    assert murder["YTD"]["months_reported"] == 3  # a reported zero counts as a reported month
    assert murder["YTD"]["quarters_reported"] == 2
    assert murder["M03"]["incident_count"] is None
    assert murder["YTD"]["offense_bucket"] == "violent"
    assert murder["YTD"]["agency_name_normalized"] == "alpha police"
    robbery_ytd = next(r for r in rows if r["offense_slug"] == "robbery" and r["period"] == "YTD")
    assert robbery_ytd["incident_count"] is None and robbery_ytd["months_reported"] == 0
    assert robbery_ytd["quarters_reported"] is None
    buckets = {r["offense_slug"]: r["offense_bucket"] for r in rows}
    assert buckets["motor_vehicle_theft"] == "property" and buckets["arson"] == "other"
    prostitution = next(r for r in rows if r["offense_name_source"] == "Prostitutition")
    assert prostitution["offense_name"] == "Prostitution" and prostitution["offense_slug"] == "prostitution"


def test_parser_requires_the_year_sheet(tmp_path) -> None:
    path = tmp_path / "fibrs.xlsx"
    _synthetic_workbook(path, 2025)
    with pytest.raises(KeyError):
        parse_fdle_fibrs_offense_detail(path, 2024)


@lru_cache(maxsize=None)
def _parsed_0826(year: int) -> tuple:
    return tuple(parse_fdle_fibrs_offense_detail(WORKBOOK_0826, year))


@pytest.mark.skipif(not WORKBOOK_0826.exists(), reason="FDLE 08-2026 workbook not staged")
@pytest.mark.parametrize(
    ("year", "agency", "slug", "expected", "agencies"),
    [
        (2024, "Alachua County Sheriff's Office", "robbery", 24, 225),
        (2024, "Miami-Dade Police Department", "larceny_shoplifting", 4734, 225),
        (2025, "Miami-Dade Sheriff's Office", "robbery", 637, 254),
        (2025, "Alachua County Sheriff's Office", "prostitution", 3, 254),
    ],
)
def test_known_rows_from_08_2026_workbook(year, agency, slug, expected, agencies) -> None:
    rows = _parsed_0826(year)
    assert len({r["source_row_number"] for r in rows}) == agencies
    ytd = [r for r in rows if r["agency_name"] == agency and r["offense_slug"] == slug and r["period"] == "YTD"]
    assert len(ytd) == 1
    assert ytd[0]["incident_count"] == expected
    monthly = [r["incident_count"] for r in rows if r["agency_name"] == agency and r["offense_slug"] == slug and r["period"] != "YTD"]
    assert sum(monthly) == expected


def test_the_miami_dade_sheriffs_office_matches_the_former_police_department():
    from crimerisk.state_publications import normalize_fdle_publication_match_name

    assert normalize_fdle_publication_match_name("Miami-Dade Sheriff's Office") == (
        normalize_fdle_publication_match_name("Miami-Dade Police Department")
    )


def test_a_zero_filled_month_of_an_active_agency_is_not_a_reported_month(tmp_path) -> None:
    """The 08-2026 workbook writes 0 into unreported months; an agency large enough to log
    something every month that logs nothing at all in a month did not report it."""
    year = 2024
    wb = Workbook()
    ws = wb.active
    ws.title = f"FIBRS Offense {year}"
    ws.append([f"{year} FIBRS Reported Offenses by Offense Type and Month"])
    offense_row = [None, None, f" {year} Population Estimate"]
    month_row = ["County", "Agency Name", None]
    for name in OFFENSES:
        offense_row += [name] + [None] * 11
        month_row += [f"{year} {m}" for m in MONTH_NAMES]
    ws.append(offense_row)
    ws.append(month_row)
    # Q4 only: zeros January-September in every block, real counts October-December.
    q4_larceny = [0] * 9 + [40, 35, 38]
    q4_burglary = [0] * 9 + [5, 4, 6]
    blocks = []
    for name in OFFENSES:
        blocks += q4_larceny if name == "Larceny - All Other" else q4_burglary if name == "Burglary" else [0] * 12
    ws.append(["Broward", "Q4 Police Department", 175000] + blocks)
    # A tiny agency: one offense in the year, zeros elsewhere -- an all-zero month says nothing.
    tiny = []
    for name in OFFENSES:
        tiny += [0] * 11 + [1] if name == "Burglary" else [0] * 12
    ws.append(["Liberty", "Tiny Police Department", 900] + tiny)
    path = tmp_path / "fibrs.xlsx"
    wb.save(path)
    rows = parse_fdle_fibrs_offense_detail(path, year)
    ytd = {(r["agency_name"], r["offense_slug"]): r for r in rows if r["period"] == "YTD"}
    q4 = ytd[("Q4 Police Department", "larceny_all_other")]
    assert q4["months_reported"] == 3 and q4["quarters_reported"] == 1 and q4["incident_count"] == 113
    assert ytd[("Q4 Police Department", "robbery")]["months_reported"] == 3
    assert ytd[("Tiny Police Department", "burglary")]["months_reported"] == 12
