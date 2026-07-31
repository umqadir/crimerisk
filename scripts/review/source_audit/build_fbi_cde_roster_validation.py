from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _pick_roster_path(base_dir: Path) -> Path:
    parsed_dir = base_dir / "parsed"
    full = parsed_dir / "agency_rosters_2024.parquet"
    priority = parsed_dir / "agency_rosters_priority_states_2024.parquet"
    if full.exists():
        return full
    if priority.exists():
        return priority
    raise FileNotFoundError(f"No FBI/CDE roster parquet found under {base_dir}")


def _dominant_string(s: pd.Series) -> str | None:
    vals = [str(v) for v in s.dropna().astype(str) if str(v)]
    if not vals:
        return None
    counts = pd.Series(vals).value_counts()
    return str(counts.index[0])


def build_validation(*, repo_root: Path, roster_parquet: Path | None = None, out_dir: Path | None = None) -> dict[str, str]:
    base_dir = repo_root / "data" / "FBI-CDE-Agency-Rosters-2024"
    roster_path = roster_parquet or _pick_roster_path(base_dir)
    validation_dir = (
        out_dir
        or (
            repo_root
            / "state"
            / "review"
            / "analysis"
            / "source_audit"
            / "fbi_cde_roster_validation"
        )
    ).resolve()
    validation_dir.mkdir(parents=True, exist_ok=True)

    roster = pd.read_parquet(roster_path).copy()
    roster["ori9"] = roster["ori"].astype("string").str.slice(0, 9)
    roster["state_abbr"] = roster["state_abbr"].astype("string")
    roster["official_is_nibrs"] = roster["is_nibrs"].fillna(False).astype(bool)
    roster["official_nibrs_start_date"] = roster["nibrs_start_date"].astype("string")
    roster["official_nibrs_start_year"] = pd.to_numeric(
        roster["official_nibrs_start_date"].str.slice(0, 4), errors="coerce"
    ).astype("Int64")
    roster["official_effective_nibrs_2024"] = (
        roster["official_is_nibrs"]
        & roster["official_nibrs_start_year"].fillna(9999).le(2024)
    )
    roster_keep = [
        "ori9",
        "state_abbr",
        "agency_name",
        "agency_type_name",
        "official_is_nibrs",
        "official_nibrs_start_date",
        "official_nibrs_start_year",
        "official_effective_nibrs_2024",
    ]
    roster = roster[roster_keep].drop_duplicates("ori9").copy()

    agency_master = pd.read_parquet(repo_root / "state" / "reference" / "agency_master.parquet").copy()
    agency_master["ori9"] = agency_master["ori9"].astype("string")
    agency_master["state_abbr"] = agency_master["state_abbr"].astype("string")
    agency_keep = [
        "ori9",
        "state_abbr",
        "state_fips",
        "county_fips",
        "place_fips",
        "agency_name_std",
        "agency_type_norm",
        "source_presence_srs",
        "source_presence_nibrs",
        "latest_srs_year",
        "latest_nibrs_year",
    ]
    agency_master = agency_master[agency_keep].drop_duplicates("ori9").copy()

    obs = pd.read_parquet(repo_root / "state" / "observations" / "agency_year_observations.parquet").copy()
    obs = obs[obs["year"].eq(2024)].copy()
    obs["ori9"] = obs["ori9"].astype("string")
    obs["source"] = obs["source"].astype("string")
    obs["count"] = pd.to_numeric(obs["count"], errors="coerce").fillna(0.0)
    obs_summary = (
        obs.groupby(["ori9", "source"], dropna=False)["count"].sum().reset_index()
        .pivot(index="ori9", columns="source", values="count")
        .reset_index()
    )
    obs_summary.columns.name = None
    obs_summary["repo_has_cius_2024"] = pd.to_numeric(
        obs_summary.get("cius_publication_annual", 0.0), errors="coerce"
    ).fillna(0.0).gt(0)
    obs_summary["repo_has_srs_2024"] = pd.to_numeric(
        obs_summary.get("srs_return_a_annual", 0.0), errors="coerce"
    ).fillna(0.0).gt(0)
    obs_summary["repo_has_nibrs_2024"] = pd.to_numeric(
        obs_summary.get("nibrs_srs_equivalent_annual", 0.0), errors="coerce"
    ).fillna(0.0).gt(0)

    regimes = pd.read_parquet(repo_root / "state" / "modeling" / "agency_year_reporting_regimes.parquet").copy()
    regimes = regimes[regimes["year"].eq(2024)].copy()
    regimes["ori9"] = regimes["ori9"].astype("string")
    regimes["preferred_count_for_offense"] = 0.0
    cius_mask = regimes["preferred_source_by_regime"].eq("cius_publication_annual")
    srs_mask = regimes["preferred_source_by_regime"].eq("srs_return_a_annual")
    nibrs_mask = regimes["preferred_source_by_regime"].eq("nibrs_srs_equivalent_annual")
    regimes.loc[cius_mask, "preferred_count_for_offense"] = pd.to_numeric(
        regimes.loc[cius_mask, "cius_count"], errors="coerce"
    ).fillna(0.0)
    regimes.loc[srs_mask, "preferred_count_for_offense"] = pd.to_numeric(
        regimes.loc[srs_mask, "srs_count"], errors="coerce"
    ).fillna(0.0)
    regimes.loc[nibrs_mask, "preferred_count_for_offense"] = pd.to_numeric(
        regimes.loc[nibrs_mask, "nibrs_count"], errors="coerce"
    ).fillna(0.0)
    regime_summary = (
        regimes.groupby("ori9", dropna=False)
        .agg(
            repo_preferred_source_any_offense_2024=("preferred_source_by_regime", _dominant_string),
            repo_reporting_regime_mix_2024=("reporting_regime", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
            repo_preferred_support_count_2024=("preferred_count_for_offense", "sum"),
            repo_cius_regime_offense_count_2024=("preferred_source_by_regime", lambda s: int((s == "cius_publication_annual").sum())),
            repo_nibrs_regime_offense_count_2024=("preferred_source_by_regime", lambda s: int((s == "nibrs_srs_equivalent_annual").sum())),
            repo_srs_regime_offense_count_2024=("preferred_source_by_regime", lambda s: int((s == "srs_return_a_annual").sum())),
        )
        .reset_index()
    )

    merged = (
        roster.merge(agency_master, on=["ori9", "state_abbr"], how="left")
        .merge(obs_summary, on="ori9", how="left")
        .merge(regime_summary, on="ori9", how="left")
    )
    merged["repo_has_srs_2024"] = pd.Series(merged["repo_has_srs_2024"], dtype="boolean").fillna(False).astype(bool)
    merged["repo_has_nibrs_2024"] = pd.Series(merged["repo_has_nibrs_2024"], dtype="boolean").fillna(False).astype(bool)
    merged["repo_present_in_master"] = merged["agency_name_std"].notna()
    merged["repo_preferred_source_any_offense_2024"] = merged["repo_preferred_source_any_offense_2024"].astype("string")
    merged["repo_reporting_regime_mix_2024"] = merged["repo_reporting_regime_mix_2024"].astype("string")
    merged["repo_preferred_support_count_2024"] = pd.to_numeric(
        merged["repo_preferred_support_count_2024"], errors="coerce"
    ).fillna(0.0)

    presence_flag = pd.Series("aligned", index=merged.index, dtype="string")
    presence_flag.loc[~merged["repo_present_in_master"]] = "missing_in_repo_master"
    presence_flag.loc[
        merged["repo_present_in_master"] & ~merged["repo_has_srs_2024"] & ~merged["repo_has_nibrs_2024"]
    ] = "present_but_no_2024_observation"
    merged["official_repo_presence_flag"] = presence_flag

    transition_flag = pd.Series("aligned_or_mixed", index=merged.index, dtype="string")
    transition_flag.loc[
        merged["official_is_nibrs"]
        & merged["official_nibrs_start_year"].fillna(9999).gt(2024)
    ] = "official_nibrs_start_after_2024"
    transition_flag.loc[
        merged["official_effective_nibrs_2024"] & ~merged["repo_has_nibrs_2024"]
    ] = "official_nibrs_but_repo_no_nibrs_2024"
    transition_flag.loc[
        (~merged["official_is_nibrs"]) & merged["repo_has_nibrs_2024"]
    ] = "repo_nibrs_but_official_not_nibrs"
    transition_flag.loc[
        merged["official_is_nibrs"]
        & merged["official_nibrs_start_year"].fillna(9999).le(2024)
        & ~merged["repo_has_nibrs_2024"]
    ] = "official_nibrs_started_by_2024_but_repo_no_nibrs"
    transition_flag.loc[
        merged["official_is_nibrs"]
        & merged["repo_has_nibrs_2024"]
        & merged["repo_has_srs_2024"]
    ] = "mixed_source_2024"
    merged["official_repo_transition_flag"] = transition_flag

    reasons: list[str] = []
    for row in merged.itertuples(index=False):
        row_reasons: list[str] = []
        if not bool(row.repo_present_in_master):
            row_reasons.append("missing_in_repo_master")
        if bool(row.official_effective_nibrs_2024) and not bool(row.repo_has_nibrs_2024):
            row_reasons.append("official_nibrs_no_repo_nibrs")
        if (not bool(row.official_is_nibrs)) and bool(row.repo_has_nibrs_2024):
            row_reasons.append("repo_nibrs_without_official_nibrs")
        if pd.notna(row.official_nibrs_start_year) and int(row.official_nibrs_start_year) <= 2024 and not bool(row.repo_has_nibrs_2024):
            row_reasons.append("official_start_by_2024_but_repo_no_nibrs")
        if bool(row.repo_present_in_master) and not bool(row.repo_has_srs_2024) and not bool(row.repo_has_nibrs_2024):
            row_reasons.append("no_2024_observation")
        reasons.append("|".join(row_reasons) if row_reasons else "aligned")
    merged["mismatch_reason"] = pd.Series(reasons, dtype="string")

    validation_cols = [
        "ori9",
        "state_abbr",
        "state_fips",
        "county_fips",
        "place_fips",
        "agency_name",
        "agency_type_name",
        "agency_name_std",
        "agency_type_norm",
        "official_is_nibrs",
        "official_effective_nibrs_2024",
        "official_nibrs_start_date",
        "official_nibrs_start_year",
        "repo_present_in_master",
        "repo_has_srs_2024",
        "repo_has_nibrs_2024",
        "repo_preferred_source_any_offense_2024",
        "repo_reporting_regime_mix_2024",
        "repo_preferred_support_count_2024",
        "repo_nibrs_regime_offense_count_2024",
        "repo_srs_regime_offense_count_2024",
        "official_repo_presence_flag",
        "official_repo_transition_flag",
        "mismatch_reason",
    ]
    validation = merged[validation_cols].sort_values(
        ["state_abbr", "official_is_nibrs", "repo_preferred_support_count_2024", "ori9"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    priority = validation[validation["mismatch_reason"].ne("aligned")].copy()
    priority = priority.sort_values(
        ["repo_preferred_support_count_2024", "state_abbr", "ori9"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    state_summary = (
        validation.assign(
            mismatch_flag=validation["mismatch_reason"].ne("aligned"),
            official_nibrs_flag=validation["official_is_nibrs"].fillna(False),
        )
        .groupby("state_abbr", dropna=False)
        .agg(
            official_oris=("ori9", "nunique"),
            official_nibrs_oris=("official_nibrs_flag", "sum"),
            official_effective_nibrs_oris=("official_effective_nibrs_2024", "sum"),
            repo_master_matches=("repo_present_in_master", "sum"),
            mismatch_rows=("mismatch_flag", "sum"),
            mismatch_weight=("repo_preferred_support_count_2024", lambda s: float(s[validation.loc[s.index, "mismatch_reason"].ne("aligned")].sum())),
            official_nibrs_repo_no_nibrs=("official_repo_transition_flag", lambda s: int((s == "official_nibrs_but_repo_no_nibrs_2024").sum() + (s == "official_nibrs_started_by_2024_but_repo_no_nibrs").sum())),
            repo_nibrs_official_false=("official_repo_transition_flag", lambda s: int((s == "repo_nibrs_but_official_not_nibrs").sum())),
            no_2024_observation=("official_repo_presence_flag", lambda s: int((s == "present_but_no_2024_observation").sum())),
        )
        .reset_index()
        .sort_values(["mismatch_weight", "mismatch_rows", "state_abbr"], ascending=[False, False, True], kind="mergesort")
    )

    validation_parquet = validation_dir / "ori_source_validation_2024.parquet"
    validation_csv = validation_dir / "ori_source_validation_2024.csv"
    summary_csv = validation_dir / "state_source_transition_summary_2024.csv"
    priority_csv = validation_dir / "priority_transition_mismatches_2024.csv"
    validation.to_parquet(validation_parquet, index=False)
    validation.to_csv(validation_csv, index=False)
    state_summary.to_csv(summary_csv, index=False)
    priority.to_csv(priority_csv, index=False)
    return {
        "roster_path": str(roster_path),
        "validation_parquet": str(validation_parquet),
        "validation_csv": str(validation_csv),
        "summary_csv": str(summary_csv),
        "priority_csv": str(priority_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ORI-level FBI/CDE official source validation artifacts.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--roster-parquet", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    result = build_validation(
        repo_root=args.repo_root.resolve(),
        roster_parquet=args.roster_parquet.resolve() if args.roster_parquet is not None else None,
        out_dir=args.out_dir.resolve() if args.out_dir is not None else None,
    )
    print(result)


if __name__ == "__main__":
    main()
