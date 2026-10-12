"""Read-only DuckDB query over a packet's tables (the reviewer's one command).

    uv run python scripts/agents/query.py --packet <packet_dir> "SELECT ..."

Every .parquet and .csv file directly in the packet directory is a view named after its stem.
The connection is in-memory; nothing is written. At most 200 rows are printed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("sql")
    args = parser.parse_args()
    packet = (REPO_ROOT / args.packet).resolve() if not Path(args.packet).is_absolute() else Path(args.packet).resolve()
    if not packet.is_relative_to((REPO_ROOT / "state" / "review").resolve()) and not packet.is_relative_to((REPO_ROOT / "analysis_scratch").resolve()):
        raise SystemExit("packet must be under state/review/ or analysis_scratch/")
    if re.search(r"\b(copy|attach|install|load|export|create|insert|update|delete|drop|alter|pragma|set)\b", args.sql, re.I):
        raise SystemExit("read-only: SELECT / WITH / DESCRIBE / SUMMARIZE only")
    con = duckdb.connect(":memory:")
    for path in sorted(packet.iterdir()):
        name = re.sub(r"\W", "_", path.stem)
        if path.suffix == ".parquet":
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
        elif path.suffix == ".csv":
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{path}', all_varchar=true)")
    print(con.execute(args.sql).fetchdf().head(200).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
