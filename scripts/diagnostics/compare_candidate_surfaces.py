"""Compare numerical, publication, tail, and conservation properties of candidate surfaces."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


OFFENSES = (
    "murder", "rape", "robbery", "aggravated_assault", "burglary", "larceny",
    "motor_vehicle_theft",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arms(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    arms = payload["arms"] if isinstance(payload, dict) else payload
    if not arms or len({row["name"] for row in arms}) != len(arms):
        raise ValueError("arms must have unique names and list frozen first")
    return arms


def _service_transfer_path(arm: dict[str, object], manifest: dict[str, object]) -> Path | None:
    explicit = arm.get("service_state_transfer_path")
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    declared = summary.get("service_state_transfer_path") if isinstance(summary, dict) else None
    value = explicit or declared
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(str(arm["manifest_path"])).parent / path.name
    return path


def _controls_path(
    arm: dict[str, object], manifest: dict[str, object]
) -> tuple[Path, Path | None]:
    """Resolve a hash-bound, arm-owned control snapshot."""
    manifest_path = Path(str(arm["manifest_path"]))
    year = int(manifest["year"])
    relative = Path("controls") / f"jurisdiction_controls_smoothed_{year}.parquet"
    frozen_manifest_path = manifest_path.parent / "frozen_upstream" / "manifest.json"
    inventory_sha: str | None = None
    snapshot_path: Path | None = None
    if frozen_manifest_path.exists():
        inventory = json.loads(frozen_manifest_path.read_text())
        rows = [
            row
            for row in inventory.get("files", [])
            if row.get("path") == relative.as_posix()
        ]
        if len(rows) != 1 or not rows[0].get("sha256"):
            raise ValueError(f"frozen inventory must bind exactly one {relative}")
        inventory_sha = str(rows[0]["sha256"])
        snapshot_path = frozen_manifest_path.parent / relative

    explicit = arm.get("controls_path")
    path = Path(str(explicit)) if explicit else snapshot_path
    if path is None:
        raise ValueError(
            f"{arm['name']} requires controls_path and controls_sha256 "
            "without a frozen inventory"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    actual_sha = _sha256(path)
    expected_sha = arm.get("controls_sha256")
    if expected_sha and actual_sha != str(expected_sha):
        raise ValueError(f"{arm['name']} controls_path does not match controls_sha256")
    if inventory_sha and actual_sha != inventory_sha:
        raise ValueError(f"{arm['name']} controls_path does not match frozen inventory")
    if not inventory_sha and not expected_sha:
        raise ValueError(f"{arm['name']} requires controls_sha256 for legacy controls")

    input_stats = manifest.get("input_file_stats")
    recorded = (
        input_stats.get("jurisdiction_controls_smoothed")
        if isinstance(input_stats, dict)
        else None
    )
    if not isinstance(recorded, dict) or not bool(recorded.get("exists")):
        raise ValueError(f"{arm['name']} manifest does not record smoothed controls")
    if int(recorded.get("size_bytes", -1)) != path.stat().st_size:
        raise ValueError(f"{arm['name']} controls snapshot differs from recorded input size")
    snapshot_mtime = datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    if str(recorded.get("mtime_utc")) != snapshot_mtime:
        raise ValueError(f"{arm['name']} controls snapshot differs from recorded input mtime")
    return path, frozen_manifest_path if frozen_manifest_path.exists() else None


def _transfer_net_by_state(ledger: pd.DataFrame) -> pd.Series:
    """Net geographic imports, indexed by allocation state and offense."""
    if ledger.empty:
        return pd.Series(dtype=float)
    required = {
        "service_scope_id", "canonical_target_ori", "offense", "source_state_fips",
        "allocation_state_fips", "source_target_count", "allocation_count",
        "route_reason",
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"service transfer ledger missing columns {missing}")
    work = ledger.copy()
    work["source_state_fips"] = work["source_state_fips"].astype("string").str.zfill(2)
    work["allocation_state_fips"] = work["allocation_state_fips"].astype("string").str.zfill(2)
    located = work[
        work["allocation_state_fips"].notna()
        & work["allocation_state_fips"].astype("string").str.strip().ne("")
    ]
    incoming = located.groupby(["allocation_state_fips", "offense"])[
        "allocation_count"
    ].sum()
    outgoing = located.groupby(["source_state_fips", "offense"])[
        "allocation_count"
    ].sum()
    incoming.index = incoming.index.set_names(["state_fips", "offense"])
    outgoing.index = outgoing.index.set_names(["state_fips", "offense"])
    return incoming.subtract(outgoing, fill_value=0.0)


def _control_change_authorized(
    arm: dict[str, object], *, manifest_text: str
) -> tuple[bool, str | None]:
    """Verify an explicit control-change exception against one hash-bound build input."""
    contract = arm.get("control_change_contract")
    if not isinstance(contract, dict) or not bool(contract.get("allowed")):
        return False, None
    path_value = contract.get("input_path")
    expected_sha = contract.get("input_sha256")
    if not path_value or not expected_sha:
        raise ValueError("control_change_contract requires input_path and input_sha256")
    path = Path(str(path_value)).resolve()
    if not path.exists() or _sha256(path) != str(expected_sha):
        raise ValueError(f"control_change_contract input hash does not match {path}")
    if str(path) not in manifest_text:
        raise ValueError(f"control_change_contract input {path} is not bound in the arm manifest")
    return True, str(contract.get("reason") or "intentional_scope_change")


def _numeric_summary(
    frame: pd.DataFrame, *, arm: str, support: str, id_col: str, column: str
) -> dict[str, object]:
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    valid = values[finite]
    maximum_index = valid.idxmax() if not valid.empty else None
    return {
        "arm": arm, "support": support, "column": column, "rows": len(frame),
        "null_count": int(values.isna().sum()), "nonfinite_count": int((~finite & values.notna()).sum()),
        "negative_count": int(values.lt(0.0).sum()), "zero_count": int(values.eq(0.0).sum()),
        "sum": float(valid.sum()), "p99": float(valid.quantile(0.99)) if len(valid) else None,
        "p999": float(valid.quantile(0.999)) if len(valid) else None,
        "max": float(valid.max()) if len(valid) else None,
        "max_support_id": str(frame.loc[maximum_index, id_col]) if maximum_index is not None else None,
        "max_jurisdiction_id": (
            str(frame.loc[maximum_index, "eb_jurisdiction_id"])
            if maximum_index is not None and "eb_jurisdiction_id" in frame else None
        ),
    }


def _load(path: Path, support: str) -> tuple[pd.DataFrame, str]:
    id_col = "block_group_geoid" if support == "block_group" else "tract_id"
    schema = pq.ParquetFile(path).schema_arrow.names
    population_columns = [column for column in schema if column.startswith("population_")]
    fixed = [id_col, "state_fips", "tract_id", "eb_jurisdiction_id", *population_columns]
    prefixes = ("expected_count_", "primary_denominator_", "index_publishable_", "primary_index_publishable_")
    composites = {"index_event_burden_resident", "index_harm_burden_resident"}
    columns = list(dict.fromkeys(
        [column for column in fixed if column in schema]
        + [column for column in schema if column.startswith(prefixes) or column in composites]
    ))
    frame = pd.read_parquet(path, columns=columns)
    frame[id_col] = frame[id_col].astype("string")
    if frame[id_col].duplicated().any():
        raise ValueError(f"duplicate {support} identifiers in {path}")
    return frame, id_col


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()
    arms = _arms(args.arms_json)
    inputs = [args.arms_json]
    unlocated_paths: dict[str, Path] = {}
    manifests: dict[str, dict[str, object]] = {}
    manifest_texts: dict[str, str] = {}
    transfer_paths: dict[str, Path | None] = {}
    control_paths: dict[str, Path] = {}
    for arm in arms:
        for key in ("block_group_path", "tract_path", "manifest_path", "audit_path"):
            if key not in arm or not arm[key]:
                raise ValueError(f"{arm['name']} requires {key}")
            path = Path(str(arm[key]))
            if not path.exists():
                raise FileNotFoundError(path)
            inputs.append(path)
        name = str(arm["name"])
        manifest_path = Path(str(arm["manifest_path"]))
        manifest_text = manifest_path.read_text()
        manifest = json.loads(manifest_text)
        manifests[name] = manifest
        manifest_texts[name] = manifest_text
        unlocated_path = Path(str(arm.get("unlocated_mass_path") or (
            Path(str(arm["manifest_path"])).parent
            / f"unlocated_mass_{int(manifest['year'])}.parquet"
        )))
        if not unlocated_path.exists():
            raise FileNotFoundError(unlocated_path)
        unlocated_paths[name] = unlocated_path
        inputs.append(unlocated_path)
        transfer_path = _service_transfer_path(arm, manifest)
        if transfer_path is not None:
            if not transfer_path.exists():
                raise FileNotFoundError(transfer_path)
            inputs.append(transfer_path)
        transfer_paths[name] = transfer_path
        control_path, frozen_manifest_path = _controls_path(arm, manifest)
        control_paths[name] = control_path
        inputs.append(control_path)
        if frozen_manifest_path is not None:
            inputs.append(frozen_manifest_path)
    loaded: dict[tuple[str, str], tuple[pd.DataFrame, str]] = {}
    for arm in arms:
        loaded[(str(arm["name"]), "block_group")] = _load(Path(str(arm["block_group_path"])), "block_group")
        loaded[(str(arm["name"]), "tract")] = _load(Path(str(arm["tract_path"])), "tract")

    frozen_name = str(arms[0]["name"])
    summaries: list[dict[str, object]] = []
    extremes: list[pd.DataFrame] = []
    comparisons: list[dict[str, object]] = []
    publication_population: list[dict[str, object]] = []
    for arm in arms:
        name = str(arm["name"])
        for support in ("block_group", "tract"):
            frame, id_col = loaded[(name, support)]
            frozen, frozen_id = loaded[(frozen_name, support)]
            ids = pd.Index(frame[id_col])
            frozen_ids = pd.Index(frozen[frozen_id])
            comparisons.append({
                "arm": name, "support": support, "rows": len(frame),
                "missing_frozen_ids": int(len(frozen_ids.difference(ids))),
                "extra_ids": int(len(ids.difference(frozen_ids))),
                "duplicate_ids": int(frame[id_col].duplicated().sum()),
            })
            expected = [f"expected_count_{offense}" for offense in OFFENSES]
            denominators = [f"primary_denominator_{offense}" for offense in OFFENSES]
            composites = [
                column for column in (
                    "expected_count_personal", "expected_count_property", "expected_count_total",
                    "index_event_burden_resident", "index_harm_burden_resident",
                ) if column in frame
            ]
            for column in [*expected, *denominators, *composites]:
                if column not in frame:
                    summaries.append({
                        "arm": name, "support": support, "column": column,
                        "missing_column": True,
                    })
                    continue
                summaries.append(_numeric_summary(frame, arm=name, support=support, id_col=id_col, column=column))
                if column.startswith(("expected_count_", "index_")):
                    trace_columns = [
                        candidate for candidate in (
                            id_col, "tract_id", "state_fips", "eb_jurisdiction_id", column
                        ) if candidate in frame
                    ]
                    tail = frame.nlargest(args.top_n, column)[list(dict.fromkeys(trace_columns))].copy()
                    tail.insert(0, "support", support)
                    tail.insert(0, "arm", name)
                    tail.insert(2, "column", column)
                    extremes.append(tail)

            publication_columns = [
                column for column in frame
                if column.startswith(("index_publishable_", "primary_index_publishable_"))
            ]
            comparisons[-1]["publication_true"] = {
                column: int(frame[column].fillna(False).astype(bool).sum())
                for column in publication_columns
            }
            comparisons[-1]["publication_null"] = {
                column: int(frame[column].isna().sum()) for column in publication_columns
            }
            for column in publication_columns:
                publication_population.extend([
                    {"arm": name, "support": support, "column": column,
                     "metric": "publishable_rows", "value": int(frame[column].fillna(False).astype(bool).sum())},
                    {"arm": name, "support": support, "column": column,
                     "metric": "null_rows", "value": int(frame[column].isna().sum())},
                ])
            population_column = next(
                (column for column in frame if column.startswith("population_")), None
            )
            if population_column is not None:
                population = pd.to_numeric(frame[population_column], errors="coerce").fillna(0.0)
                comparisons[-1]["population_total"] = float(population.sum())
                comparisons[-1]["publication_true_population"] = {
                    column: float(population[frame[column].fillna(False).astype(bool)].sum())
                    for column in publication_columns
                }
                comparisons[-1]["denominator_null_population"] = {
                    column: float(population[frame[column].isna()].sum())
                    for column in denominators if column in frame
                }
                for column in publication_columns:
                    publication_population.append({
                        "arm": name, "support": support, "column": column,
                        "metric": "publishable_population",
                        "value": float(population[frame[column].fillna(False).astype(bool)].sum()),
                    })
                for column in denominators:
                    if column in frame:
                        publication_population.append({
                            "arm": name, "support": support, "column": column,
                            "metric": "null_population",
                            "value": float(population[frame[column].isna()].sum()),
                        })

    frozen_bg, _ = loaded[(frozen_name, "block_group")]
    unlocated = {
        name: pd.read_parquet(path, columns=["state_fips", "offense", "unlocated_count"])
        for name, path in unlocated_paths.items()
    }
    transfers = {
        name: pd.read_parquet(path) if path is not None else pd.DataFrame()
        for name, path in transfer_paths.items()
    }
    controls = {
        name: pd.read_parquet(
            path, columns=["year", "state_fips", "offense", "smoothed_count"]
        )
        for name, path in control_paths.items()
    }
    control_change = {
        name: _control_change_authorized(arm, manifest_text=manifest_texts[name])
        for arm in arms
        for name in [str(arm["name"])]
    }
    state_checks: list[dict[str, object]] = []
    rare_checks: list[dict[str, object]] = []
    for arm in arms:
        name = str(arm["name"])
        frame, _ = loaded[(name, "block_group")]
        tract, _ = loaded[(name, "tract")]
        transfer_net = _transfer_net_by_state(transfers[name])
        arm_controls = controls[name].copy()
        arm_controls["state_fips"] = arm_controls["state_fips"].astype("string").str.zfill(2)
        arm_controls = arm_controls[
            arm_controls["year"].astype(int).eq(int(manifests[name]["year"]))
        ]
        frozen_controls = controls[frozen_name].copy()
        frozen_controls["state_fips"] = frozen_controls["state_fips"].astype("string").str.zfill(2)
        frozen_controls = frozen_controls[
            frozen_controls["year"].astype(int).eq(int(manifests[frozen_name]["year"]))
        ]
        for offense in OFFENSES:
            column = f"expected_count_{offense}"
            current = frame.groupby("state_fips")[column].sum()
            frozen = frozen_bg.groupby("state_fips")[column].sum()
            current_unlocated = (
                unlocated[name].loc[unlocated[name]["offense"].eq(offense)]
                .groupby("state_fips")["unlocated_count"].sum()
            )
            frozen_unlocated = (
                unlocated[frozen_name].loc[unlocated[frozen_name]["offense"].eq(offense)]
                .groupby("state_fips")["unlocated_count"].sum()
            )
            located_delta = current.subtract(frozen, fill_value=0.0)
            unlocated_delta = current_unlocated.subtract(frozen_unlocated, fill_value=0.0)
            accounted_delta = located_delta.add(unlocated_delta, fill_value=0.0)
            if not transfer_net.empty and offense in transfer_net.index.get_level_values("offense"):
                net = transfer_net.xs(offense, level="offense")
            else:
                net = pd.Series(dtype=float)
            own_control = arm_controls.loc[arm_controls["offense"].eq(offense)].set_index(
                "state_fips"
            )["smoothed_count"].groupby(level=0).sum()
            frozen_control = frozen_controls.loc[
                frozen_controls["offense"].eq(offense)
            ].set_index("state_fips")["smoothed_count"].groupby(level=0).sum()
            source_accounted = current.add(current_unlocated, fill_value=0.0).subtract(
                net, fill_value=0.0
            )
            own_delta = source_accounted.subtract(own_control, fill_value=0.0)
            control_delta = own_control.subtract(frozen_control, fill_value=0.0)
            authorized, reason = control_change[name]
            state_checks.append({
                "arm": name, "offense": offense,
                "max_state_absolute_located_delta_vs_frozen": float(located_delta.abs().max()),
                "national_located_delta_vs_frozen": float(current.sum() - frozen.sum()),
                "max_state_absolute_unlocated_delta_vs_frozen": float(unlocated_delta.abs().max()),
                "national_unlocated_delta_vs_frozen": float(
                    current_unlocated.sum() - frozen_unlocated.sum()
                ),
                "max_state_absolute_accounted_delta_vs_frozen": float(accounted_delta.abs().max()),
                "national_accounted_delta_vs_frozen": float(accounted_delta.sum()),
                "max_state_absolute_transfer_net": float(net.abs().max()) if len(net) else 0.0,
                "national_transfer_net": float(net.sum()) if len(net) else 0.0,
                "max_state_absolute_own_source_control_delta": float(own_delta.abs().max()),
                "national_own_source_control_delta": float(own_delta.sum()),
                "max_state_absolute_source_control_delta_vs_frozen": float(control_delta.abs().max()),
                "national_source_control_delta_vs_frozen": float(control_delta.sum()),
                "control_change_authorized": authorized,
                "control_change_reason": reason,
            })
        for offense in ("murder", "rape"):
            column = f"expected_count_{offense}"
            bg_tract = frame.groupby("tract_id")[column].sum()
            published = tract.set_index("tract_id")[column]
            delta = bg_tract.subtract(published, fill_value=0.0)
            rare_checks.append({
                "arm": name, "offense": offense,
                "max_bg_to_tract_absolute_delta": float(delta.abs().max()),
                "national_bg_to_tract_delta": float(bg_tract.sum() - published.sum()),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    numeric_summary = pd.DataFrame(summaries)
    frozen_numeric = numeric_summary[numeric_summary["arm"].eq(frozen_name)][
        ["support", "column", "sum", "null_count"]
    ].rename(columns={"sum": "frozen_sum", "null_count": "frozen_null_count"})
    numeric_summary = numeric_summary.merge(
        frozen_numeric, on=["support", "column"], how="left", validate="many_to_one"
    )
    numeric_summary["sum_delta_vs_frozen"] = numeric_summary["sum"] - numeric_summary["frozen_sum"]
    numeric_summary["null_count_delta_vs_frozen"] = (
        numeric_summary["null_count"] - numeric_summary["frozen_null_count"]
    )
    numeric_summary.to_csv(args.out_dir / "numeric_summary.csv", index=False)
    pd.DataFrame(state_checks).to_csv(args.out_dir / "state_offense_conservation.csv", index=False)
    pd.DataFrame(rare_checks).to_csv(args.out_dir / "rare_support_identity.csv", index=False)
    pd.concat(extremes, ignore_index=True).to_csv(args.out_dir / "extreme_cells.csv", index=False)
    publication_frame = pd.DataFrame(publication_population)
    frozen_publication = publication_frame[publication_frame["arm"].eq(frozen_name)][
        ["support", "column", "metric", "value"]
    ].rename(columns={"value": "frozen_value"})
    publication_frame = publication_frame.merge(
        frozen_publication, on=["support", "column", "metric"], how="left", validate="many_to_one"
    )
    publication_frame["delta_vs_frozen"] = publication_frame["value"] - publication_frame["frozen_value"]
    publication_frame.to_csv(args.out_dir / "publication_population.csv", index=False)
    output = {
        "version": "candidate_surface_comparison_v1",
        "frozen_arm": frozen_name, "arms": [str(row["name"]) for row in arms],
        "inputs": {str(path.resolve()): _sha256(path) for path in inputs},
        "surface_comparisons": comparisons,
        "failures": {
            "geometry": int(sum(row["missing_frozen_ids"] + row["extra_ids"] for row in comparisons)),
            "nonfinite": int(sum(row.get("nonfinite_count", 0) for row in summaries)),
            "negative": int(sum(row.get("negative_count", 0) for row in summaries)),
            "own_source_control_reconciliation": int(sum(
                row["max_state_absolute_own_source_control_delta"] > 1e-6
                for row in state_checks
            )),
            "unauthorized_source_control_change": int(sum(
                row["max_state_absolute_source_control_delta_vs_frozen"] > 1e-6
                and not row["control_change_authorized"]
                for row in state_checks
            )),
            "rare_support_identity": int(sum(row["max_bg_to_tract_absolute_delta"] > 1e-6 for row in rare_checks)),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["failures"], indent=2, sort_keys=True))
    return 1 if any(output["failures"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
