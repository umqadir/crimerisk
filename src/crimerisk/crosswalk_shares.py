from __future__ import annotations

import numpy as np
import pandas as pd


def _bg_key_col(df: pd.DataFrame) -> str:
    for col in ("block_group_geoid", "bg_id"):
        if col in df.columns:
            return col
    raise ValueError("Block-group crosswalk must include block_group_geoid or bg_id")


def preferred_allocation_share(bg_crosswalk: pd.DataFrame) -> pd.Series:
    bg = bg_crosswalk.copy()
    if "state_fips" in bg.columns:
        bg["state_fips"] = bg["state_fips"].astype("string").str.zfill(2)
    key_col = _bg_key_col(bg)
    bg[key_col] = bg[key_col].astype("string").str.zfill(12)

    chosen = pd.Series(np.nan, index=bg.index, dtype=float)
    for col in ["pop_share", "housing_share", "block_share", "aland_share", "allocation_share"]:
        if col not in bg.columns:
            continue
        share = pd.to_numeric(bg.get(col), errors="coerce").fillna(0.0).clip(lower=0.0)
        chosen = chosen.where(~(chosen.isna() & share.gt(0)), share)
    return pd.to_numeric(chosen, errors="coerce").fillna(0.0).clip(lower=0.0)


def normalize_block_group_allocation_shares(bg_crosswalk: pd.DataFrame) -> pd.DataFrame:
    bg = bg_crosswalk.copy()
    if "state_fips" in bg.columns:
        bg["state_fips"] = bg["state_fips"].astype("string").str.zfill(2)
    key_col = _bg_key_col(bg)
    bg[key_col] = bg[key_col].astype("string").str.zfill(12)
    bg["allocation_share"] = preferred_allocation_share(bg)

    totals = bg.groupby(["state_fips", key_col], dropna=False)["allocation_share"].transform("sum")
    counts = bg.groupby(["state_fips", key_col], dropna=False)["allocation_share"].transform("size")
    bg["allocation_share"] = np.where(
        pd.to_numeric(totals, errors="coerce").fillna(0.0) > 0,
        pd.to_numeric(bg["allocation_share"], errors="coerce").fillna(0.0)
        / pd.to_numeric(totals, errors="coerce").fillna(1.0),
        np.where(
            pd.to_numeric(counts, errors="coerce").fillna(0.0) > 0,
            1.0 / pd.to_numeric(counts, errors="coerce").fillna(1.0),
            0.0,
        ),
    )
    return bg
