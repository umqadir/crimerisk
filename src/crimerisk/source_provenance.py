"""Source taxonomy and provenance labels for the annual observation panel.

Terminology is governed by docs/FBI-DATA-GUIDE.md. Key facts behind the labels:

* ``srs_return_a_annual`` rows come from the FBI Return A master file as released.
  Post-2020 the majority of those rows are FBI-converted NIBRS, not native SRS
  submissions (86% of 2024 local agencies). To distinguish converted from native rows,
  test NIBRS batch-header membership for the ORI-year; the Return A file itself carries
  no flag. Their conversion status is therefore ``as_released_return_a`` — "value as
  released in the Return A master file", deliberately not "native SRS".
* ``nibrs_srs_equivalent_annual`` rows are this repo's own rollup of the raw NIBRS
  segments using the FBI's documented SRS scoring rules (per-victim person crimes,
  rape = 11A+11B+11C, per-vehicle MVT, hotel rule; see crime/nibrs.py).
"""

from __future__ import annotations

import pandas as pd


CIUS_SOURCE = "cius_publication_annual"
LOCAL_PUBLICATION_SOURCE = "local_publication_annual"
STATE_PUBLICATION_SOURCE = "state_publication_annual"
SUMMARY_SOURCE = "srs_return_a_annual"
NIBRS_SOURCE = "nibrs_srs_equivalent_annual"

SUMMARY_FAMILY = "summary_annual"
NIBRS_FAMILY = "nibrs_rollup_annual"

CIUS_ORIGIN = "cius_local"
LOCAL_PUBLICATION_ORIGIN = "local_publication"
STATE_PUBLICATION_ORIGIN = "state_publication"
SRS_ORIGIN = "srs"
NIBRS_ROLLUP_ORIGIN = "nibrs_rollup"

CIUS_DATA_SOURCE = "fbi_cius_local_publication"
LOCAL_PUBLICATION_DATA_SOURCE = "local_official_publication"
STATE_PUBLICATION_DATA_SOURCE = "state_official_publication"
SRS_DATA_SOURCE = "kaplan_openicpsr_srs_return_a"
NIBRS_DATA_SOURCE = "kaplan_fbi_nibrs_master_segments"

SOURCE_PRIORITY = (
    CIUS_SOURCE,
    LOCAL_PUBLICATION_SOURCE,
    STATE_PUBLICATION_SOURCE,
    SUMMARY_SOURCE,
    NIBRS_SOURCE,
)


def source_family_from_source(source: object) -> str:
    if str(source) in {CIUS_SOURCE, LOCAL_PUBLICATION_SOURCE, STATE_PUBLICATION_SOURCE, SUMMARY_SOURCE}:
        return SUMMARY_FAMILY
    if str(source) == NIBRS_SOURCE:
        return NIBRS_FAMILY
    return "unknown"


def source_origin_from_parts(
    *,
    source: object,
    conversion_status: object | None = None,
    cius_reference_flag: object | None = None,
) -> str:
    source_text = str(source)
    if source_text == CIUS_SOURCE:
        return CIUS_ORIGIN
    if source_text == LOCAL_PUBLICATION_SOURCE:
        return LOCAL_PUBLICATION_ORIGIN
    if source_text == STATE_PUBLICATION_SOURCE:
        return STATE_PUBLICATION_ORIGIN
    if source_text == SUMMARY_SOURCE:
        return SRS_ORIGIN
    if source_text == NIBRS_SOURCE:
        return NIBRS_ROLLUP_ORIGIN
    return "unknown"


def source_lane_from_source(source: object) -> str:
    if str(source) == CIUS_SOURCE:
        return "reported_cius_publication"
    if str(source) == LOCAL_PUBLICATION_SOURCE:
        return "reported_local_publication"
    if str(source) == STATE_PUBLICATION_SOURCE:
        return "reported_state_publication"
    if str(source) == SUMMARY_SOURCE:
        return "reported_srs_summary"
    if str(source) == NIBRS_SOURCE:
        return "reported_nibrs_rollup"
    return "reported_other"


def raw_data_source_from_parts(
    *,
    source: object,
    conversion_status: object | None = None,
    cius_reference_flag: object | None = None,
) -> str:
    source_text = str(source)
    if source_text == CIUS_SOURCE:
        return CIUS_DATA_SOURCE
    if source_text == LOCAL_PUBLICATION_SOURCE:
        return LOCAL_PUBLICATION_DATA_SOURCE
    if source_text == STATE_PUBLICATION_SOURCE:
        return STATE_PUBLICATION_DATA_SOURCE
    if source_text == SUMMARY_SOURCE:
        return SRS_DATA_SOURCE
    if source_text == NIBRS_SOURCE:
        return NIBRS_DATA_SOURCE
    return "unknown"


def reporting_mode_from_source(source: object) -> str:
    if str(source) in {CIUS_SOURCE, LOCAL_PUBLICATION_SOURCE, STATE_PUBLICATION_SOURCE, SUMMARY_SOURCE}:
        return "annual_summary"
    if str(source) == NIBRS_SOURCE:
        return "annual_srs_equivalent_rollup"
    return "other"


def default_conversion_status_from_source(source: object) -> str:
    if str(source) == CIUS_SOURCE:
        return "published_summary_reference"
    if str(source) == LOCAL_PUBLICATION_SOURCE:
        return "published_local_reference"
    if str(source) == STATE_PUBLICATION_SOURCE:
        return "published_state_reference"
    if str(source) == SUMMARY_SOURCE:
        return "as_released_return_a"
    if str(source) == NIBRS_SOURCE:
        return "srs_rules_rollup_from_nibrs"
    return "other"


def initialize_preferred_source(
    *,
    has_cius: pd.Series,
    has_local_publication: pd.Series | None = None,
    has_state_publication: pd.Series,
    has_srs: pd.Series,
    has_nibrs: pd.Series,
    prefer_nibrs_mask: pd.Series | None = None,
) -> pd.Series:
    if has_local_publication is None:
        has_local_publication = pd.Series(False, index=has_cius.index)
    else:
        has_local_publication = has_local_publication.fillna(False).astype(bool)
    preferred = pd.Series(pd.NA, index=has_cius.index, dtype="string")
    preferred.loc[has_cius] = CIUS_SOURCE
    preferred.loc[~has_cius & has_local_publication] = LOCAL_PUBLICATION_SOURCE
    preferred.loc[~has_cius & ~has_local_publication & has_state_publication] = STATE_PUBLICATION_SOURCE
    preferred.loc[~has_cius & ~has_local_publication & ~has_state_publication & has_srs] = SUMMARY_SOURCE
    preferred.loc[~has_cius & ~has_local_publication & ~has_state_publication & ~has_srs & has_nibrs] = NIBRS_SOURCE
    if prefer_nibrs_mask is not None:
        preferred.loc[prefer_nibrs_mask.astype(bool)] = NIBRS_SOURCE
    return preferred


def build_prefer_nibrs_mask(
    *,
    has_cius: pd.Series,
    has_local_publication: pd.Series | None = None,
    has_state_publication: pd.Series,
    has_srs: pd.Series,
    has_nibrs: pd.Series,
    regime_prefers_nibrs: pd.Series | None = None,
    srs_regime_inferior: pd.Series | None = None,
    nibrs_supports_better: pd.Series | None = None,
    srs_count_num: pd.Series | None = None,
    nibrs_months: pd.Series | None = None,
    manual_source_override: pd.Series | None = None,
    published_nibrs_supports_nibrs: pd.Series | None = None,
) -> pd.Series:
    index = has_cius.index

    def _false_mask(mask: pd.Series | None) -> pd.Series:
        if mask is None:
            return pd.Series(False, index=index)
        return mask.fillna(False).astype(bool)

    regime_prefers_nibrs = _false_mask(regime_prefers_nibrs)
    srs_regime_inferior = _false_mask(srs_regime_inferior)
    nibrs_supports_better = _false_mask(nibrs_supports_better)
    manual_source_override = _false_mask(manual_source_override)
    published_nibrs_supports_nibrs = _false_mask(published_nibrs_supports_nibrs)
    if srs_count_num is None:
        srs_count_num = pd.Series(0.0, index=index)
    else:
        srs_count_num = pd.to_numeric(srs_count_num, errors="coerce").fillna(0.0)
    if has_local_publication is None:
        has_local_publication = pd.Series(False, index=index)
    else:
        has_local_publication = has_local_publication.fillna(False).astype(bool)
    if nibrs_months is None:
        nibrs_months = pd.Series(0.0, index=index)
    else:
        nibrs_months = pd.to_numeric(nibrs_months, errors="coerce").fillna(0.0)

    return ~has_cius & ~has_local_publication & ~has_state_publication & has_nibrs & (
        ~has_srs
        | manual_source_override
        | regime_prefers_nibrs
        | published_nibrs_supports_nibrs
        | (has_srs & srs_regime_inferior & nibrs_supports_better)
        | (has_srs & srs_count_num.eq(0) & nibrs_months.gt(0))
    )


def assign_preferred_value(
    frame: pd.DataFrame,
    *,
    output_col: str,
    preferred_source_col: str,
    source_to_input_col: dict[str, str],
    default_source: str = NIBRS_SOURCE,
) -> pd.DataFrame:
    out = frame.copy()
    out[output_col] = out[source_to_input_col[default_source]]
    for source, input_col in source_to_input_col.items():
        if source == default_source:
            continue
        mask = out[preferred_source_col].eq(source)
        out.loc[mask, output_col] = out.loc[mask, input_col]
    return out
