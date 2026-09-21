"""Generate docs/FIELDS.md from the public schema in frontend/build/crschema.py.

The public download schema is defined once, in crschema. This script reads it
and writes one table row per published column.

Run:  uv run python docs/make_fields.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "frontend" / "build"))

from crschema import (  # noqa: E402
    COMPOSITES,
    OFFENSE_LABEL,
    OFFENSES,
    PUBLIC_COMPOSITE_NAMES,
    PUBLIC_OFFENSE_FIELDS,
    TRACT_ONLY_OFFENSES,
    YEAR,
)

OUT = REPO / "docs" / "FIELDS.md"

BOTH = "Block group and tract"
TRACT = "Tract"
BG = "Block group"

# name -> (meaning, unit, support)
IDENTIFIERS = [
    ("block_group_geoid", "2020 Census block group identifier, 12 digits.", "Identifier", BG),
    ("tract_id", "2020 Census tract identifier, 11 digits.", "Identifier", BOTH),
    ("state_fips", "State FIPS code, two digits.", "Identifier", BOTH),
    ("state", "State postal abbreviation, two letters.", "Identifier", BOTH),
    (f"population_{YEAR}", f"Resident population, {YEAR}.", "Persons", BOTH),
    ("land_area_sq_mi", "Land area, 2020 TIGER, water excluded.", "Square miles", BOTH),
    (
        "special_use_type",
        "Area character: ordinary, park_open_space, industrial_employment, "
        "institutional_facility, campus_institution, transient_destination, "
        "group_quarters_other, unknown_special_use.",
        "Category",
        BOTH,
    ),
    ("jurisdiction_id", "Police jurisdiction whose total anchors this area.", "Identifier", BOTH),
    ("jurisdiction_name", "Display name of that police jurisdiction.", "Text", BOTH),
]

# offence field suffix -> (meaning template, unit)
OFFENSE_FIELDS = {
    "expected_count": ("Estimated {label} offenses in {year}.", "Offenses"),
    "primary_denominator": (
        "Exposure denominator for {label}: persons, premises or vehicles.",
        "Exposure units",
    ),
    "rate_primary": ("{label} offenses per 100,000 exposure units.", "Per 100,000"),
    "index_primary": ("{label} Crime exposure index.", "Index, 100 = US average"),
    "rate_resident": ("{label} offenses per 100,000 residents.", "Per 100,000"),
    "index_resident": ("{label} Per resident index.", "Index, 100 = US average"),
    "source_mode": (
        "Origin of the {label} estimate: direct_city_incident, mixed, modeled_transfer.",
        "Category",
    ),
}

COMPOSITE_MEANING = {
    "overall_crime_exposure": "All seven offenses combined, exposure denominators.",
    "violent_crime_exposure": "Murder, rape, robbery and aggravated assault, exposure denominators.",
    "property_crime_exposure": "Burglary, larceny and motor vehicle theft, exposure denominators.",
    "overall_per_resident": "All seven offenses combined, resident denominator.",
    "violent_per_resident": "Murder, rape, robbery and aggravated assault, resident denominator.",
    "property_per_resident": "Burglary, larceny and motor vehicle theft, resident denominator.",
}


def rows() -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = list(IDENTIFIERS)
    for offense in OFFENSES:
        label = OFFENSE_LABEL[offense]
        rare = offense in TRACT_ONLY_OFFENSES
        for _src, out_tpl in PUBLIC_OFFENSE_FIELDS:
            name = out_tpl.format(o=offense)
            suffix = name[len(offense) + 1 :]
            meaning_tpl, unit = OFFENSE_FIELDS[suffix]
            meaning = meaning_tpl.format(label=label, year=YEAR)
            support = BOTH
            if rare and suffix not in {"expected_count", "primary_denominator"}:
                support = TRACT
            out.append((name, meaning, unit, support))
    for key in COMPOSITES:
        for measure in ("exposure", "resident"):
            name = PUBLIC_COMPOSITE_NAMES[(key, measure)]
            out.append((name, COMPOSITE_MEANING[name], "Index, 100 = US average", BOTH))
    return out


def main() -> None:
    table = rows()
    lines = [
        "# Published fields",
        "",
        f"Generated from `frontend/build/crschema.py` by `docs/make_fields.py`. Edition {YEAR}.1.",
        "",
        f"Columns: {len(table)} in the block-group file, {len(table) - 1} in the tract file.",
        "The tract file omits `block_group_geoid`.",
        "",
        "Support is the geography the value is estimated at.",
        "Murder and rape rate and index columns are null in the block-group file.",
        "Murder and rape block-group expected counts are within-tract allocations for reconciliation.",
        "",
        "| Column | Meaning | Unit | Support |",
        "|---|---|---|---|",
    ]
    for name, meaning, unit, support in table:
        lines.append(f"| `{name}` | {meaning} | {unit} | {support} |")
    lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"{OUT}: {len(table)} columns")


if __name__ == "__main__":
    main()
