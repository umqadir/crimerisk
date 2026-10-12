from __future__ import annotations

from crimerisk.state_publications import _parse_ms_tops_csv_export

_ENVELOPE = (
    '"Property Crimes - 5-year Percent Change"\n'
    '"Current date: 10/6/2026 4:27:56 PM (Central Daylight Time)"\n'
    '"Data source: MS_FBI, Offense Data"\n'
    '"Jurisdiction by Geography: Clarksdale Police Department"\n'
    '"Measures: Number of Crimes"\n'
)


def test_ms_tops_export_without_a_2024_column_still_parses() -> None:
    # Agencies first reporting in 2025 export only the 2025-2026 columns.
    text = _ENVELOPE + (
        '"Incident Date","2026","2025 - 2026 Growth %","2025",\n'
        '"Offense Type"\n'
        '"Larceny - Theft","204","-10.13","227"\n'
        '"Burglary","131","-2.96","135"\n'
    )
    parsed = _parse_ms_tops_csv_export(text)
    assert parsed is not None
    assert parsed["jurisdiction_name"] == "Clarksdale Police Department"
    header = [cell.strip() for cell in parsed["header"]]
    assert "2024" not in header
    year_idx = header.index("2025")
    assert {row[0]: row[year_idx] for row in parsed["data_rows"]} == {"Larceny - Theft": "227", "Burglary": "135"}
