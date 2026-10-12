"""Write the M7 feature policy: the release policy with the non-racial protected groups admitted.

M7 (docs/NOTES.md, 2025.2 pre-registration addendum) admits the columns excluded only as protected
class that are not race, ethnicity, national origin, ancestry or language. Every row is copied from
`feature_transfer_policy_<year>.parquet`; only the rows of ADMITTED_GROUPS change class, so the
prior (stage 10b), the expert catalog (stage 11) and the residual model keep their default
exclusions (`between_only`, `excluded_protected`) and see these columns as admitted.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crimerisk.paths import RepoPaths

ADMITTED_GROUPS = ("sex_composition", "familial_status_composition")
ADMITTED_CLASS = "admitted_nonracial_protected"
EXPECTED_ADMITTED_COLUMNS = 26


def m7_policy(policy: pd.DataFrame) -> pd.DataFrame:
    out = policy.copy()
    admit = out["final_class"].eq("excluded_protected") & out["feature_group"].isin(ADMITTED_GROUPS)
    if int(admit.sum()) != EXPECTED_ADMITTED_COLUMNS:
        raise ValueError(
            f"expected {EXPECTED_ADMITTED_COLUMNS} excluded_protected columns in {ADMITTED_GROUPS}, "
            f"found {int(admit.sum())}"
        )
    out.loc[admit, "final_class"] = ADMITTED_CLASS
    out.loc[admit, "rationale"] = "M7 candidate: protected-class composition that is not race or national origin."
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    paths = RepoPaths.from_repo_root(REPO_ROOT)
    source = paths.state_dir / "modeling" / f"feature_transfer_policy_{args.year}.parquet"
    out_path = source.with_name(f"feature_transfer_policy_{args.year}_m7.parquet")
    out = m7_policy(pd.read_parquet(source))
    out.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out["final_class"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
