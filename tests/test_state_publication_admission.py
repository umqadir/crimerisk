"""State-publication rows that are not the FBI definition, and one organisation under one key."""

from __future__ import annotations

import pandas as pd

from crimerisk.state_publications import (
    FDLE_PUBLICATION_RAW_DATA_SOURCE,
    MS_TOPS_PUBLICATION_RAW_DATA_SOURCE,
    NY_DCJS_PUBLICATION_RAW_DATA_SOURCE,
    drop_definition_mismatches,
    ny_dcjs_agency_match_candidates,
)


def test_sex_offense_group_categories_are_not_admitted_as_rape():
    rows = pd.DataFrame({
        "raw_data_source": [FDLE_PUBLICATION_RAW_DATA_SOURCE] * 2 + [MS_TOPS_PUBLICATION_RAW_DATA_SOURCE] * 2
        + [NY_DCJS_PUBLICATION_RAW_DATA_SOURCE],
        "offense": ["rape", "robbery", "rape", "murder", "rape"],
        "count": [10.0, 5.0, 7.0, 1.0, 3.0],
    })
    kept = drop_definition_mismatches(rows)
    # FDLE forcible sex offenses and TOPS non-consensual sex offenses include fondling;
    # DCJS "Rape" is the FBI definition.
    assert list(zip(kept["raw_data_source"], kept["offense"], strict=True)) == [
        (FDLE_PUBLICATION_RAW_DATA_SOURCE, "robbery"),
        (MS_TOPS_PUBLICATION_RAW_DATA_SOURCE, "murder"),
        (NY_DCJS_PUBLICATION_RAW_DATA_SOURCE, "rape"),
    ]


def test_dcjs_county_state_police_matches_the_fbi_county_state_police_ori_first():
    tier_a, tier_b = ny_dcjs_agency_match_candidates("Albany County State Police")
    assert "sp albany county" in tier_a
    assert "albany state police" in tier_b
