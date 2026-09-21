from __future__ import annotations

import io
import zipfile

import pandas as pd

from crimerisk.fbi_state_estimates import parse_cius_table5_state_estimates
from crimerisk.fbi_publications import STATE_NAME_TO_ABBR


def test_parse_cius_table5_state_estimates(tmp_path):
    expected_abbrs = set(STATE_NAME_TO_ABBR.values()) - {"AS", "CZ", "GU", "MP", "PR", "VI"}
    rows = [
        ["Table 5"],
        ["Crime in the United States"],
        ["by State, 2025"],
        [
            "State",
            "Area",
            "Unused",
            "Population1",
            "Violent crime",
            "Murder and nonnegligent manslaughter",
            "Rape",
            "Robbery",
            "Aggravated assault",
            "Property crime",
            "Burglary",
            "Larceny-theft",
            "Motor vehicle theft",
        ],
    ]
    test_states = [
        state_name
        for state_name, state_abbr in STATE_NAME_TO_ABBR.items()
        if state_abbr in expected_abbrs
    ]
    for idx, state_name in enumerate(test_states, start=1):
        rows.extend(
            [
                [state_name.title(), "Metropolitan Statistical Area"],
                [None, "State Total", None, 1000 + idx, 100 + idx, idx, 2, 3, 95, 200, 20, 150, 30],
            ]
        )
    workbook = io.BytesIO()
    pd.DataFrame(rows).to_excel(workbook, index=False, header=False)
    zip_path = tmp_path / "cius-estimations-2025.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "CIUS_Table_5_Crime_in_the_United_States_by_State_2025.xlsx",
            workbook.getvalue(),
        )

    parsed = parse_cius_table5_state_estimates(zip_path, year=2025)

    assert len(parsed) == len(expected_abbrs) + 1
    assert parsed.iloc[0]["state_name"] == "United States Total"
    assert set(parsed["state_abbr"].dropna()) == expected_abbrs
    alabama = parsed[parsed["state_abbr"].eq("AL")].iloc[0]
    assert alabama["population"] == 1001
    assert alabama["homicide"] == 1
    assert alabama["rape_revised"] == 2
    assert alabama["motor_vehicle_theft"] == 30
