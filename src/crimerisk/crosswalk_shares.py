"""Within-block-group allocation shares for the block-group -> jurisdiction crosswalk.

`allocation_share` answers one question: of a block group's exposure, what fraction does
each jurisdiction that touches it own? Everything downstream multiplies it by the block
group's activity weight (`bg_weight x allocation_share`), so the share is not just a
bookkeeping fraction -- it is the apportionment that decides which jurisdiction's rate a
slice of a block group is painted with.

Two rules, both from the Stage 4 first-read audit (state/qa/stage4_screen, screens a6/a7):

1. **One basis per block group.** The ladder `pop_share -> housing_share -> block_share ->
   aland_share` is resolved once for the whole block group, not per row. The previous
   per-row form let a populated fragment be measured by population while a zero-population
   fragment in the same block group was measured by its block count, and then summed the two
   as if commensurable: 13,798 block groups (5.8%) normalised a mixed basis and 44,922
   counts reached 9,427 block groups through fragments with no residents at all. Under one
   basis a fragment that is zero on the block group's basis is not a recipient.

2. **A fragment below `CROSSWALK_MINIMUM_RECIPIENT_SHARE` is not an independent recipient.**
   `bg_weight x allocation_share` apportions the block group's *whole* activity weight by
   the fragment's coverage, which assumes the fragment's activity is proportional to its
   residents. Below 2% coverage the fragment is a block-assignment boundary artifact --
   median 3 census blocks carrying 10 residents, against a median block group of 22 blocks,
   so the fragment is under half of one block's share (1/22 = 4.5%) of the block group.
   Nothing in the fragment's geometry supports the uniformity assumption, while the mass it
   delivers is set by its jurisdiction's total rather than by the fragment: Lakewood CO at
   share 0.017 delivered 228 counts, 67.5% of a 1,025-person block group. Below the floor
   the fragment's share is routed to the block group's remaining recipients, and the sliver
   jurisdiction's own target spreads over the block groups it materially covers -- mass is
   conserved because the component share is normalised within the jurisdiction, not within
   the block group.

The floor **never strands mass**: it is not applied to a fragment that is the only positive
recipient left in its block group, in its jurisdiction, or in its county's non-municipal
remainder. Those three pools are the three allocation pools a zeroed share could empty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Ladder resolved once per block group, most direct measure of resident exposure first.
# `allocation_share` is the last rung: rows carrying a precomputed share and no block-level
# geometry at all (the geometry builder's municipal rescue path) enter there.
ALLOCATION_SHARE_LADDER: tuple[str, ...] = (
    "pop_share",
    "housing_share",
    "block_share",
    "aland_share",
    "allocation_share",
)
DEGENERATE_BASIS = "degenerate_equal_split"

# See rule 2 in the module docstring for the basis of this value.
CROSSWALK_MINIMUM_RECIPIENT_SHARE = 0.02

RECIPIENT = "recipient"
ZERO_ON_BASIS = "zero_on_block_group_basis"
BELOW_FLOOR = "below_minimum_recipient_share"
FLOOR_EXEMPT_BLOCK_GROUP = "floor_exempt_only_recipient_in_block_group"
FLOOR_EXEMPT_JURISDICTION = "floor_exempt_only_support_for_jurisdiction"
FLOOR_EXEMPT_COUNTY_REMAINDER = "floor_exempt_only_support_for_county_remainder"

STATE_REMAINDER_TYPE = "state_nonmunicipal_remainder"


def _bg_key_col(df: pd.DataFrame) -> str:
    for col in ("block_group_geoid", "bg_id"):
        if col in df.columns:
            return col
    raise ValueError("Block-group crosswalk must include block_group_geoid or bg_id")


def _normalized_keys(bg: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = bg.copy()
    if "state_fips" in out.columns:
        out["state_fips"] = out["state_fips"].astype("string").str.zfill(2)
    key_col = _bg_key_col(out)
    out[key_col] = out[key_col].astype("string").str.zfill(12)
    return out, key_col


def block_group_allocation_basis(bg_crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Resolve one basis per block group and return its per-row value.

    Returns a frame with `allocation_basis` (the ladder rung the whole block group is
    measured on) and `allocation_basis_value` (this row's raw value on that rung).
    """
    bg, key_col = _normalized_keys(bg_crosswalk)
    group_cols = ["state_fips", key_col] if "state_fips" in bg.columns else [key_col]
    grouper = [bg[col] for col in group_cols]

    basis = pd.Series(pd.NA, index=bg.index, dtype="string")
    value = pd.Series(np.nan, index=bg.index, dtype=float)
    for col in ALLOCATION_SHARE_LADDER:
        if col not in bg.columns:
            continue
        share = pd.to_numeric(bg[col], errors="coerce").fillna(0.0).clip(lower=0.0)
        block_group_total = share.groupby(grouper, dropna=False).transform("sum")
        take = basis.isna() & pd.to_numeric(block_group_total, errors="coerce").fillna(0.0).gt(0.0)
        basis = basis.where(~take, col)
        value = value.where(~take, share)
    return pd.DataFrame(
        {
            "allocation_basis": basis.fillna(DEGENERATE_BASIS),
            "allocation_basis_value": value.fillna(0.0),
        },
        index=bg.index,
    )


def preferred_allocation_share(bg_crosswalk: pd.DataFrame) -> pd.Series:
    """This row's value on its block group's single chosen basis (unnormalised)."""
    return block_group_allocation_basis(bg_crosswalk)["allocation_basis_value"]


def _pool_max(value: pd.Series, keys: list[pd.Series]) -> pd.Series:
    return value.groupby(keys, dropna=False).transform("max")


def normalize_block_group_allocation_shares(
    bg_crosswalk: pd.DataFrame,
    *,
    minimum_recipient_share: float = CROSSWALK_MINIMUM_RECIPIENT_SHARE,
) -> pd.DataFrame:
    """Write `allocation_share` summing to 1 per block group over one consistent basis.

    Adds three audit columns: `allocation_basis` (the block group's ladder rung),
    `allocation_share_before_recipient_floor` (the share on that basis before the sliver
    floor) and `allocation_recipient_status` (why a row is or is not a recipient).
    """
    bg, key_col = _normalized_keys(bg_crosswalk)
    if bg.empty:
        for col, default in (
            ("allocation_share", 0.0),
            ("allocation_basis", pd.NA),
            ("allocation_share_before_recipient_floor", 0.0),
            ("allocation_recipient_status", pd.NA),
        ):
            if col not in bg.columns:
                bg[col] = default
        return bg

    resolved = block_group_allocation_basis(bg)
    bg["allocation_basis"] = resolved["allocation_basis"]
    value = resolved["allocation_basis_value"]

    bg_group_cols = ["state_fips", key_col] if "state_fips" in bg.columns else [key_col]
    bg_keys = [bg[col] for col in bg_group_cols]
    totals = pd.to_numeric(value.groupby(bg_keys, dropna=False).transform("sum"), errors="coerce").fillna(0.0)
    counts = pd.to_numeric(value.groupby(bg_keys, dropna=False).transform("size"), errors="coerce").fillna(0.0)
    base_share = pd.Series(
        np.where(
            totals > 0,
            value / totals.where(totals > 0, 1.0),
            np.where(counts > 0, 1.0 / counts.where(counts > 0, 1.0), 0.0),
        ),
        index=bg.index,
        dtype=float,
    )
    bg["allocation_share_before_recipient_floor"] = base_share

    status = pd.Series(RECIPIENT, index=bg.index, dtype="string")
    status.loc[base_share.le(0.0)] = ZERO_ON_BASIS

    floor = float(minimum_recipient_share)
    if floor > 0.0:
        below = base_share.gt(0.0) & base_share.lt(floor)
        # "The floor never strands mass": exempt a sliver that is the last positive
        # recipient of any pool a zeroed share could empty.
        exempt_block_group = base_share.eq(_pool_max(base_share, bg_keys)) & base_share.gt(0.0)
        status.loc[below & exempt_block_group] = FLOOR_EXEMPT_BLOCK_GROUP

        remaining = below & ~exempt_block_group
        if bool(remaining.any()) and {"jurisdiction_id"} <= set(bg.columns):
            juris_keys = [bg[col] for col in ("state_fips", "jurisdiction_id", "jurisdiction_type") if col in bg.columns]
            exempt_juris = base_share.eq(_pool_max(base_share, juris_keys)) & base_share.gt(0.0)
            status.loc[remaining & exempt_juris] = FLOOR_EXEMPT_JURISDICTION
            remaining = remaining & ~exempt_juris

        if bool(remaining.any()) and "jurisdiction_type" in bg.columns and "state_fips" in bg.columns:
            is_remainder = bg["jurisdiction_type"].astype("string").eq(STATE_REMAINDER_TYPE)
            county = bg["state_fips"].astype("string") + bg[key_col].astype("string").str.slice(2, 5)
            remainder_share = base_share.where(is_remainder, 0.0)
            exempt_county = (
                is_remainder
                & remainder_share.gt(0.0)
                & remainder_share.eq(_pool_max(remainder_share, [county]))
            )
            status.loc[remaining & exempt_county] = FLOOR_EXEMPT_COUNTY_REMAINDER
            remaining = remaining & ~exempt_county

        status.loc[remaining] = BELOW_FLOOR

    kept = base_share.where(status.ne(BELOW_FLOOR), 0.0)
    kept_totals = pd.to_numeric(kept.groupby(bg_keys, dropna=False).transform("sum"), errors="coerce").fillna(0.0)
    bg["allocation_share"] = np.where(
        kept_totals > 0,
        kept / kept_totals.where(kept_totals > 0, 1.0),
        np.where(counts > 0, 1.0 / counts.where(counts > 0, 1.0), 0.0),
    )
    bg["allocation_recipient_status"] = status
    return bg


def assert_allocation_shares_conserve(
    bg_crosswalk: pd.DataFrame,
    *,
    tolerance: float = 1e-9,
) -> None:
    """Fail closed if the normalised shares do not sum to 1 in every block group.

    Not asserted before the Stage 4 audit; the recipient floor makes it load-bearing,
    because a floor that emptied a block group would silently lose that block group's mass.
    """
    if bg_crosswalk.empty:
        return
    bg, key_col = _normalized_keys(bg_crosswalk)
    group_cols = ["state_fips", key_col] if "state_fips" in bg.columns else [key_col]
    totals = (
        pd.to_numeric(bg["allocation_share"], errors="coerce")
        .fillna(0.0)
        .groupby([bg[col] for col in group_cols], dropna=False)
        .sum()
    )
    bad = totals[(totals - 1.0).abs() > float(tolerance)]
    if not bad.empty:
        raise ValueError(
            "Block-group allocation shares must sum to 1 per block group; "
            f"{len(bad)} block groups violate it (worst deviation "
            f"{float((bad - 1.0).abs().max()):.3g}): {bad.head(10).to_dict()}"
        )
