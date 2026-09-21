"""Parse the FBI's own Return A master file (`reta-<year>.txt`) into agency-month rows.

Why this exists
---------------
The Kaplan openICPSR restack of the Return A master stops at 2024, so the 2025
observation panel had exactly one agency-level lane -- the CIUS publication bundle --
covering the 9,027 agencies the FBI chose to print in the Table 8/9/10/11 family, every
row stamped `months_reported = 12` because the published tables carry no month field.
The master file the FBI distributes through the Crime Data Explorer carries every
agency on file, the per-month submission flag, and the per-month actual-offense counts.

Record layout
-------------
The file is fixed width with no interior newlines: a 305-character agency header
followed by twelve 590-character monthly segments, 7,385 characters plus a newline per
agency-year. The FBI ships no machine-readable record description with the download, so
the field offsets below were recovered from the data and are re-verified on every run
against an independent source (`--verify-against`, the CIUS publication rows already in
the observation panel): each offense field must reproduce the published annual count for
at least `--min-verify-share` of the agencies the two sources share.

Header offsets are 1-based and inclusive; segment offsets are 0-based within a segment.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


RECORD_BYTES = 7386          # 7,385 characters plus the newline
HEADER_CHARS = 305
SEGMENT_CHARS = 590
MONTHS = 12

# Header, 1-based inclusive.
HEADER_FIELDS: dict[str, tuple[int, int]] = {
    "state_code": (2, 3),
    "ori7": (4, 10),
    "population_group": (11, 11),
    "division": (13, 13),
    "year2": (14, 15),
    "population": (45, 53),
    "agency_name_raw": (151, 200),
}

# Monthly segment, 0-based.
SEGMENT_STATUS_SLICE = (8, 13)
STATUS_REPORTED = "55555"

FIELD_WIDTH = 5

# Actual-offense fields. Aggravated assault is filed only as its four weapon
# sub-lines (firearm / knife / other weapon / hands-fists-feet); the master carries no
# aggravated-assault total, so the four are summed, exactly as the Return A form does.
OFFENSE_OFFSETS: dict[str, tuple[int, ...]] = {
    "murder": (157,),
    "rape": (167,),
    "robbery": (182,),
    "aggravated_assault": (212, 217, 222, 227),
    "burglary": (237,),
    "larceny": (257,),
    "motor_vehicle_theft": (262,),
}

OFFENSES = tuple(OFFENSE_OFFSETS)


def _read_records(path: Path, *, member: str | None = None) -> np.ndarray:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            name = member or archive.namelist()[0]
            payload = archive.read(name)
    else:
        payload = path.read_bytes()
    if len(payload) % RECORD_BYTES:
        raise SystemExit(
            f"{path.name}: {len(payload)} bytes is not a multiple of the "
            f"{RECORD_BYTES}-byte record length; the layout has changed."
        )
    count = len(payload) // RECORD_BYTES
    return np.frombuffer(payload, dtype=np.uint8).reshape(count, RECORD_BYTES)


def _text_column(records: np.ndarray, start: int, end: int) -> np.ndarray:
    """1-based inclusive character slice of the record, as a string array."""
    block = records[:, start - 1 : end]
    return np.array(
        [bytes(row).decode("latin-1").strip() for row in block], dtype=object
    )


def _digits(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, valid) for a trailing-axis block of ASCII digits."""
    digit = block.astype(np.int16) - 48
    valid = ((digit >= 0) & (digit <= 9)).all(axis=-1)
    digit = np.where((digit >= 0) & (digit <= 9), digit, 0)
    weights = 10 ** np.arange(block.shape[-1] - 1, -1, -1, dtype=np.int64)
    return (digit * weights).sum(axis=-1), valid


def parse_return_a_master(path: Path) -> pd.DataFrame:
    records = _read_records(path)
    count = records.shape[0]

    header = {
        name: _text_column(records, start, end)
        for name, (start, end) in HEADER_FIELDS.items()
    }
    population, population_valid = _digits(records[:, 44:53])
    population = np.where(population_valid, population, np.nan).astype(float)

    segments = records[:, HEADER_CHARS : HEADER_CHARS + MONTHS * SEGMENT_CHARS].reshape(
        count, MONTHS, SEGMENT_CHARS
    )
    status = np.array(
        [
            [
                bytes(segments[i, m, SEGMENT_STATUS_SLICE[0] : SEGMENT_STATUS_SLICE[1]])
                .decode("latin-1")
                for m in range(MONTHS)
            ]
            for i in range(count)
        ],
        dtype=object,
    )
    reported = status == STATUS_REPORTED

    year_text = header["year2"]
    years = np.array(
        [2000 + int(value) if value.isdigit() else -1 for value in year_text]
    )

    frames: list[pd.DataFrame] = []
    for offense, offsets in OFFENSE_OFFSETS.items():
        total = np.zeros((count, MONTHS), dtype=np.int64)
        valid = np.ones((count, MONTHS), dtype=bool)
        for offset in offsets:
            values, ok = _digits(segments[:, :, offset : offset + FIELD_WIDTH])
            total = total + np.where(ok, values, 0)
            valid = valid & ok
        frames.append(
            pd.DataFrame(
                {
                    "ori7": np.repeat(header["ori7"], MONTHS),
                    "year": np.repeat(years, MONTHS),
                    "month": np.tile(np.arange(1, MONTHS + 1), count),
                    "offense": offense,
                    "count": total.reshape(-1).astype(float),
                    "field_valid": valid.reshape(-1),
                    "reported": reported.reshape(-1).astype(bool),
                    "population": np.repeat(population, MONTHS),
                    "agency_name_raw": np.repeat(header["agency_name_raw"], MONTHS),
                    "population_group": np.repeat(header["population_group"], MONTHS),
                }
            )
        )
    monthly = pd.concat(frames, ignore_index=True)
    # A count filed against a month the agency did not submit is not evidence; the
    # master carries none today and the assertion keeps it that way.
    stray = monthly.loc[~monthly["reported"], "count"].sum()
    if stray:
        raise SystemExit(
            f"{path.name}: {stray:,.0f} offenses sit in months flagged unreported; "
            "the status-code offset is wrong."
        )
    return monthly


def collapse_to_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    reported = monthly[monthly["reported"]]
    # A handful of ORIs (the "HP:" highway-patrol district aggregates, 41 of them in
    # 2025) carry several records under one identifier. Their counts add -- they are
    # different sub-units of the same reporting ORI -- but their months do not: the
    # agency-year covered twelve months whichever record filed them.
    annual = (
        reported.groupby(["ori7", "year", "offense"], dropna=False)
        .agg(
            count=("count", "sum"),
            months_reported=("month", "nunique"),
            field_valid=("field_valid", "all"),
        )
        .reset_index()
    )
    if annual["months_reported"].gt(12).any():
        raise SystemExit("months_reported exceeded twelve; the month axis is wrong.")
    meta = (
        monthly.groupby(["ori7", "year"], dropna=False)
        .agg(
            population=("population", "first"),
            agency_name_raw=("agency_name_raw", "first"),
            population_group=("population_group", "first"),
        )
        .reset_index()
    )
    annual = annual.merge(meta, on=["ori7", "year"], how="left")
    annual["months_missing"] = (12.0 - annual["months_reported"]).clip(0.0, 11.0)
    return annual


def _verify(annual: pd.DataFrame, *, reference: pd.DataFrame) -> dict[str, object]:
    """Agree with an independent publication of the same year, offense by offense."""
    merged = annual.merge(
        reference.rename(columns={"count": "reference_count"}),
        on=["ori7", "year", "offense"],
        how="inner",
    )
    out: dict[str, object] = {"compared_rows": int(len(merged))}
    per_offense: dict[str, dict[str, float]] = {}
    for offense, group in merged.groupby("offense"):
        exact = float((group["count"] - group["reference_count"]).abs().le(0.5).mean())
        per_offense[str(offense)] = {
            "rows": int(len(group)),
            "exact_share": round(exact, 4),
            "median_abs_diff": float(
                (group["count"] - group["reference_count"]).abs().median()
            ),
            "parsed_total": float(group["count"].sum()),
            "reference_total": float(group["reference_count"].sum()),
        }
    out["per_offense"] = per_offense
    out["min_exact_share"] = (
        min(v["exact_share"] for v in per_offense.values()) if per_offense else 0.0
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse the FBI Return A master file into agency-month rows."
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=REPO_ROOT / "state" / "observations" / "agency_year_observations.parquet",
        help="Observation panel used to re-verify the recovered field offsets.",
    )
    parser.add_argument("--min-verify-share", type=float, default=0.90)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    source = (
        args.source
        if args.source is not None
        else REPO_ROOT / "data" / f"FBI-UCR-Master-{args.year}" / "raw" / f"reta-{args.year}.zip"
    ).resolve()
    if not source.exists():
        raise SystemExit(f"Return A master not found: {source}")

    # The raw master directories are shared, immutable inputs; parsed products go to a
    # dated directory of their own.
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else REPO_ROOT / "data" / f"FBI-UCR-Return-A-Parsed-{args.year}-{stamp}"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly = parse_return_a_master(source)
    monthly = monthly[monthly["year"].eq(int(args.year))].copy()
    annual = collapse_to_annual(monthly)

    verification: dict[str, object] = {"performed": False}
    if not args.skip_verify and args.verify_against.exists():
        panel = pd.read_parquet(
            args.verify_against, columns=["ori7", "year", "offense", "count", "source"]
        )
        reference = panel[
            panel["year"].eq(int(args.year))
            & panel["source"].eq("cius_publication_annual")
        ][["ori7", "year", "offense", "count"]]
        if not reference.empty:
            verification = _verify(annual, reference=reference)
            verification["performed"] = True
            verification["reference_source"] = "cius_publication_annual"
            if float(verification["min_exact_share"]) < args.min_verify_share:
                print(json.dumps(verification, indent=2))
                raise SystemExit(
                    "Return A field offsets no longer reproduce the published CIUS "
                    f"counts (min exact share {verification['min_exact_share']:.3f} < "
                    f"{args.min_verify_share:.3f})."
                )

    monthly_path = out_dir / f"return_a_master_monthly_{args.year}.parquet"
    annual_path = out_dir / f"return_a_master_annual_{args.year}.parquet"
    monthly.to_parquet(monthly_path, index=False)
    annual.to_parquet(annual_path, index=False)

    reported_months = annual.groupby("ori7")["months_reported"].max()
    summary = {
        "year": int(args.year),
        "parsed_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "agency_records": int(annual["ori7"].nunique()),
        "agencies_any_month": int((reported_months >= 1).sum()),
        "agencies_full_year": int((reported_months == 12).sum()),
        "annual_rows": int(len(annual)),
        "annual_rows_any_month": int((annual["months_reported"] >= 1).sum()),
        "part1_total_reported_months": float(annual["count"].sum()),
        "verification": verification,
    }
    (out_dir / f"return_a_master_{args.year}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
