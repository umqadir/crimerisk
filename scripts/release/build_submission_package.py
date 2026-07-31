from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from generate_submission_materials import generate_submission_materials


REPO_ROOT = Path(__file__).resolve().parents[2]
MANUAL_INPUT_INVENTORY_MD = (
    REPO_ROOT / "scripts" / "release" / "assets" / "Manual_Input_Inventory.md"
)
MANUAL_INPUT_INVENTORY_CSV = (
    REPO_ROOT / "scripts" / "release" / "assets" / "manual_input_inventory.csv"
)

REPRO_RELATIVE_PATHS = [
    Path(".python-version"),
    Path("main.py"),
    Path("pyproject.toml"),
    Path("sitecustomize.py"),
    Path("uv.lock"),
    Path("src/crimerisk"),
    Path("docs/STATE.md"),
    Path("docs/SPEC.md"),
    Path("docs/AGS-CrimeRisk-Methodology-2025B.txt"),
    Path("docs/FBI-DATA-GUIDE.md"),
    Path("configs/agency_master_supplement.csv"),
    Path("configs/city_incident_categories"),
    Path("configs/city_incident_priority.csv"),
    Path("configs/city_incident_sources.csv"),
    Path("configs/consolidated_agency_detector_exceptions.csv"),
    Path("configs/consolidated_agency_footprints.csv"),
    Path("configs/county_to_cbsa_2023.csv"),
    Path("configs/local_resolution_overrides.csv"),
    Path("configs/municipal_geometry_overrides.csv"),
    Path("configs/overlap_custom_footprints.csv"),
    Path("configs/overlap_footprint_overrides.csv"),
    Path("configs/reporting_regime_overrides.csv"),
    Path("configs/source_preference_overrides.csv"),
    Path("scripts/diagnostics/qa_build.py"),
    Path("scripts/diagnostics/validate_release_outputs.py"),
    Path("scripts/diagnostics/check_promoted_allocator_inputs.py"),
    Path("scripts/diagnostics/audit_external_surface_availability.py"),
    Path("scripts/release/promote_candidate.py"),
    Path("scripts/release/assets/DATA_SOURCES_ATTRIBUTION.md"),
    Path("scripts/release/assets/METHODOLOGY_EXCLUSIONS.md"),
    Path("scripts/release/assets/data_sources_attribution.csv"),
    Path("scripts/release/assets/methodology_exclusions.json"),
    Path("state/reference/agency_master.parquet"),
    Path("state/reference/agency_to_jurisdiction_crosswalk.parquet"),
    Path("state/observations/agency_year_observations.parquet"),
    Path("state/controls/jurisdiction_controls_2024.parquet"),
    Path("state/controls/jurisdiction_year_estimates.parquet"),
    Path("state/controls/state_control_comparison.parquet"),
    Path("state/review/queues/local_resolution"),
    Path("state/review/packets/municipal_targets"),
    Path("state/review/packets/city"),
    Path("state/cache/cpi/CPIAUCSL.csv"),
    Path("state/modeling/agency_year_reporting_regimes.parquet"),
    Path("state/modeling/promoted_next_phase_allocator_preflight_2024.json"),
    Path("state/modeling/next_phase_measurement_summary_2024.json"),
    Path("state/modeling/external_surface_availability_2024.json"),
    Path("state/modeling/next_phase_validation_city_incident_share_surface_2024.parquet"),
    Path("state/modeling/county_anchoring_v4_closure_metrics.csv"),
    Path("state/modeling/county_anchoring_v4_closure_report.md"),
    Path("state/modeling/burglary_gradient_split_v4.csv"),
    Path("state/modeling/burglary_gradient_split_v2.csv"),
    Path("state/modeling/burglary_gradient_split_v3.csv"),
    Path("state/modeling/burglary_exposure_duplicate_feature_classification_2024.csv"),
    Path("state/modeling/burglary_exposure_duplicate_feature_classification_2024.json"),
    Path("data/Census-PopEst-2020-2025/co-est2025-alldata.csv"),
    Path("data/FBI-NIBRS-Tables-2024/parsed/nibrs_offense_type_by_agency_2024.parquet"),
    Path("data/tiger_bg/parsed/bg_centroids.parquet"),
    Path("data/Overture-Places/parsed/block_group_overture_places_states_latest.parquet"),
    Path("data/Overture-Places/parsed/block_group_overture_commercial_core_states_latest.parquet"),
    Path("state/output/allocation_component_denominator_audit_2024.parquet"),
    Path("state/output/city_posterior_diagnostics_2024.parquet"),
    Path("state/output/crimerisk_block_group_2024_ags_core.parquet"),
    Path("state/output/crimerisk_block_group_2024_fbi_calibrated.parquet"),
    Path("state/output/crimerisk_tract_2024_ags_core.parquet"),
    Path("state/output/crimerisk_tract_2024_fbi_calibrated.parquet"),
    Path("state/output/denominator_publishability_audit_2024.csv"),
    Path("state/output/crimerisk_output_build_2024.json"),
    Path("state/output/validation_summary.json"),
    Path("state/output/zero_target_denominator_audit_2024.parquet"),
    Path("state/qa/build_qa_summary.json"),
]

REFERENCE_COPY_PATHS = {
    Path("docs/STATE.md"): Path("references/STATE.md"),
    Path("docs/SPEC.md"): Path("references/SPEC.md"),
    Path("docs/AGS-CrimeRisk-Methodology-2025B.txt"): Path("references/AGS-CrimeRisk-Methodology-2025B.txt"),
    Path("docs/AGS-CrimeRisk-Methodology-2026A.pdf"): Path("references/AGS-CrimeRisk-Methodology-2026A.pdf"),
    Path("docs/FBI-DATA-GUIDE.md"): Path("references/FBI-DATA-GUIDE.md"),
    Path("scripts/release/assets/DATA_SOURCES_ATTRIBUTION.md"): Path("references/DATA_SOURCES_ATTRIBUTION.md"),
    Path("scripts/release/assets/METHODOLOGY_EXCLUSIONS.md"): Path("references/METHODOLOGY_EXCLUSIONS.md"),
}

VALIDATION_COPY_PATHS = {
    Path("state/qa/build_qa_summary.json"): Path("validation/build_qa_summary.json"),
    Path("state/modeling/jurisdiction_model_benchmark_2024.json"): Path("validation/jurisdiction_model_benchmark_2024.json"),
    Path("state/modeling/city_share_benchmark_2024.json"): Path("validation/city_share_benchmark_2024.json"),
    Path("state/modeling/city_residual_benchmark_2024.json"): Path("validation/city_residual_benchmark_2024.json"),
    Path("state/modeling/next_phase_city_residual_benchmark_overture_core_2024.json"): Path("validation/next_phase_city_residual_benchmark_overture_core_2024.json"),
    Path("state/modeling/next_phase_measurement_summary_2024.json"): Path("validation/next_phase_measurement_summary_2024.json"),
    Path("state/modeling/promoted_next_phase_allocator_preflight_2024.json"): Path("validation/promoted_next_phase_allocator_preflight_2024.json"),
    Path("state/modeling/dashboard_neighborhood_check_lookup_2024.json"): Path("validation/dashboard_neighborhood_check_lookup_2024.json"),
    Path("state/modeling/external_surface_availability_2024.json"): Path("validation/external_surface_availability_2024.json"),
    Path("state/output/crimerisk_output_build_2024.json"): Path("validation/crimerisk_output_build_2024.json"),
    Path("state/output/validation_summary.json"): Path("validation/validation_summary.json"),
    Path("state/output/denominator_publishability_audit_2024.csv"): Path("validation/denominator_publishability_audit_2024.csv"),
    Path("state/modeling/county_anchoring_v4_closure_metrics.csv"): Path("validation/county_anchoring_v4_closure_metrics.csv"),
    Path("state/modeling/county_anchoring_v4_closure_report.md"): Path("validation/county_anchoring_v4_closure_report.md"),
    Path("state/modeling/burglary_gradient_split_v4.csv"): Path("validation/burglary_gradient_split_v4.csv"),
    Path("state/modeling/burglary_gradient_split_v2.csv"): Path("validation/burglary_gradient_split_v2.csv"),
    Path("state/modeling/burglary_gradient_split_v3.csv"): Path("validation/burglary_gradient_split_v3.csv"),
    Path("state/modeling/burglary_exposure_duplicate_feature_classification_2024.csv"): Path("validation/burglary_exposure_duplicate_feature_classification_2024.csv"),
    Path("state/modeling/burglary_exposure_duplicate_feature_classification_2024.json"): Path("validation/burglary_exposure_duplicate_feature_classification_2024.json"),
    Path("scripts/release/assets/data_sources_attribution.csv"): Path("validation/data_sources_attribution.csv"),
    Path("scripts/release/assets/methodology_exclusions.json"): Path("validation/methodology_exclusions.json"),
    Path("state/modeling/city_calibration_benchmark_2024.json"): Path("validation/city_calibration_benchmark_2024.json"),
}

PACKAGE_FILTERED_TREES = {
    Path("state/review/packets/municipal_targets"): (
        "*/packet_manifest.json",
        "*/recommendation.csv",
        "*/published_reference_extract.csv",
        "*/*_offense_extract.csv",
    ),
    Path("state/review/packets/city"): (
        "*/packet_status.csv",
        "*/packet_offense_status.csv",
        "*/offense_crosswalk.csv",
        "*/published_reference_extract.csv",
        "*/reconciliation_summary.csv",
    ),
}


def _copy_filtered_tree(src: Path, dst: Path, *, patterns: tuple[str, ...]) -> None:
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        rel_src = src.relative_to(REPO_ROOT)
        patterns = PACKAGE_FILTERED_TREES.get(rel_src)
        if patterns is not None:
            _copy_filtered_tree(src, dst, patterns=patterns)
            return
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    else:
        shutil.copy2(src, dst)


def _prune_package_noise(out_dir: Path) -> None:
    for path in sorted(out_dir.rglob("*")):
        if path.name not in {".venv", "__pycache__", ".DS_Store"} and not (
            path.name.endswith(".egg-info") or path.name.endswith(".pyc")
        ):
            continue
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for pattern in ("*.egg-info", "*.pyc"):
        for path in sorted(out_dir.rglob(pattern)):
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _remove_existing_package(out_dir: Path) -> None:
    def _onexc(func, path, exc):
        try:
            os.chmod(path, 0o700)
            func(path)
        except OSError:
            raise exc

    for attempt in range(3):
        try:
            shutil.rmtree(out_dir, onexc=_onexc)
            return
        except OSError:
            if attempt == 2:
                subprocess.run(["/bin/rm", "-rf", str(out_dir)], check=True)
                return
            time.sleep(0.5)


def _write_zip_archive(out_dir: Path) -> Path:
    zip_path = Path(f"{out_dir}.zip")
    if zip_path.exists():
        zip_path.unlink()
    fixed_date = (2026, 7, 2, 0, 0, 0)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_dir():
                continue
            rel_path = path.relative_to(out_dir.parent).as_posix()
            info = zipfile.ZipInfo(rel_path, fixed_date)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with path.open("rb") as handle:
                zf.writestr(info, handle.read())
    return zip_path


def _write_package_readme(out_dir: Path) -> None:
    text = """# CrimeRisk 2024 Submission Package

This package contains the promoted 2024 CrimeRisk public-data release: modeled annual
expected counts, count-derived rates and indexes, reliability metadata, source-mode metadata,
and release audit sidecars at 2020 Census block-group and tract level for the 48 contiguous
states plus DC. It is an independent public-data alternative inspired by AGS CrimeRisk, not
AGS CrimeRisk and not a claim of proprietary benchmark parity.

## Released Data

The primary surface is `ags_core` under `repro/state/output/`:

- `crimerisk_block_group_2024_ags_core.parquet`
- `crimerisk_tract_2024_ags_core.parquet`

Two FBI-calibrated diagnostic surfaces are also included because the promoted build emitted
them:

- `crimerisk_block_group_2024_fbi_calibrated.parquet`
- `crimerisk_tract_2024_fbi_calibrated.parquet`

Use `index_total_primary_event_weighted` as the default total-crime index. The resident
burden view is `index_total_part1_resident`. Column definitions are summarized in
`OUTPUT_SCHEMA.md`; `materials/tables/output_schema.csv` is generated from the actual
promoted parquet columns.

## Shipped Estimator

The release allocates official 2024 jurisdiction controls to block groups, then rolls block
groups to tracts. Non-municipal and eligible state-police/highway-patrol/county-overlap lanes
are county-anchored. Current-year missing rows use trend-scaled fills where direct reporting is
not usable. City incident feeds update within-jurisdiction shares through posterior city shares,
not hard total overrides. Published rates and indexes are count-derived from expected counts and
stored denominators; diagnostic empirical-Bayes fields never overwrite the published point.

Burglary uses a premise denominator with calibrated commercial exposure, residualizes duplicate
commercial/activity exposure from the within-share model, and passes the regime-aware burglary
commercial gate. Person-exposure offenses use residents/workers plus LandScan USA modeled daytime
activity where larger, and apply the publication floor documented in
`references/METHODOLOGY_EXCLUSIONS.md`; suppressed rows keep expected counts and crime-density
fields for reconciliation.

## Validation

The promoted manifest is `repro/state/output/crimerisk_output_build_2024.json`; the release
validator summary is `repro/state/output/validation_summary.json` and is also copied to
`validation/validation_summary.json`. The promoted validator is green (`ok=true`, zero issues).
The manifest records the copied artifact hashes, the candidate validation summary hash, and the
frontend snapshot hash check.

Quick check after extracting:

```bash
cd repro
uv sync
uv run python scripts/diagnostics/validate_release_outputs.py
```

## Package Contents

- `report/CrimeRisk_Submission_Report.md` — methodology, evidence, results, and limitations
- `materials/` — regenerated tables, charts, and maps from `repro/state/output`
- `references/` — canonical docs and source/methodology reference files
- `validation/` — compact release, source, gate, and attribution evidence
- `repro/` — code, configs, reviewed inputs, promoted outputs, and rebuild contract

## Limitations

State reporting intensity and source-selection differences still affect control totals. Outside
the covered-city truth set, neighborhood texture is modeled transfer raked to official totals.
LandScan USA does not include tourists or other transitory visitors, so visitor-heavy places can
still overstate per-person risk and are flagged. Burglary modeled-transfer commercial
concentration passed the release gate but remains above the pooled covered-city truth estimate.
Kansas remains about 14% below the CDE estimate pending agency-gap review. Florida totals are
deliberately FDLE-based in this release and should not be forced to FBI CDE totals without a
source-contract decision.

These outputs are modeled aggregate research surfaces, not individual victimization
probabilities and not a basis for high-stakes individual, housing, lending, employment,
policing, or insurance decisions without independent validation and legal/ethical review.

## Rebuilding

The package omits raw public downloads. To rebuild, place the required files under
`repro/data/` at the exact paths in `materials/tables/required_inputs.csv`, then run:

```bash
cd repro
uv run python main.py build-input-manifest
uv run python main.py build-release --emit-fbi-calibrated
uv run python scripts/diagnostics/qa_build.py
```

Source attribution and acquisition notes are in `references/DATA_SOURCES_ATTRIBUTION.md`,
`validation/data_sources_attribution.csv`, `materials/tables/required_inputs.csv`, and
`references/FBI-DATA-GUIDE.md`.
"""
    (out_dir / "README.md").write_text(text)


def _write_repro_readme(repro_root: Path) -> None:
    text = """# CrimeRisk Replication — Reproducibility Bundle

A self-contained snapshot of the CrimeRisk replication pipeline: source code
(`src/crimerisk/`), configuration (`configs/`), reviewed analyst inputs
(`state/review/`), and the released outputs (`state/output/`).

Commands, run from this directory after `uv sync`:

- `uv run python scripts/diagnostics/validate_release_outputs.py` — check the shipped
  outputs (reads only files in this package)
- `uv run python scripts/diagnostics/check_promoted_allocator_inputs.py` — preflight the
  promoted residual allocator inputs shipped with this package
- `uv run python main.py build-input-manifest` — report every expected raw input and
  anything missing under `data/`
- `uv run python main.py build-release --emit-fbi-calibrated` — full rebuild; needs the
  raw source data under `data/` (see `../README.md` and
  `../materials/tables/required_inputs.csv`)
- `uv run python scripts/diagnostics/qa_build.py` — end-to-end pipeline checks after a
  rebuild

`main.py --help` lists the individual pipeline stages if you want to run them one at a
time. The `benchmark-suite` command needs benchmark scripts that are part of the
development workspace, not this package; it says so and exits if invoked here.

The promoted residual allocator support artifacts under `state/modeling/` and
`data/Overture-Places/parsed/` are included so the shipped output manifest and
output validator remain self-contained. The promoted manifest, release validator,
county-anchoring closure evidence, burglary gate evidence, attribution inventory, and
methodology-exclusion assets are copied under `validation/` for quick review without
reading the full reproducibility tree.
"""
    (repro_root / "README.md").write_text(text)


def _write_output_schema_readme(out_dir: Path) -> None:
    text = """# Output Schema

The released Parquet files contain one row per 2020 Census geography:

- `crimerisk_block_group_2024_ags_core.parquet`: one row per released 2020 block group.
- `crimerisk_tract_2024_ags_core.parquet`: one row per released 2020 tract.
- `crimerisk_block_group_2024_fbi_calibrated.parquet`: derivative block-group calibration surface.
- `crimerisk_tract_2024_fbi_calibrated.parquet`: derivative tract calibration surface.

Release coverage is the 49-state product scope: the 48 contiguous states plus the District of
Columbia. Alaska (`02`), Hawaii (`15`), Puerto Rico (`72`), and the other territories (`60`, `66`,
`69`, `78`) are excluded. Alaska and Hawaii are excluded instead of being shipped as populated
zero-count rows.

`ags_core` is the primary public-data surface. `fbi_calibrated` keeps the same within-state spatial
allocation and applies diagnostic state/offense calibration ratios for comparison with the FBI CDE
estimated-total surface. Treat `ags_core` as the shipped public-data release surface.

`expected_count_*` fields are annual modeled 2024 counts. Each offense has one primary denominator across every
released geography: person exposure for murder, rape, robbery, aggravated assault, and
larceny; occupied households plus calibrated Overture commercial-premise exposure for burglary; and
aggregate vehicles for motor vehicle theft. Person exposure is the BG-level maximum of the jobs-based
proxy (`max(residents, residents + workplace jobs - resident workers)`) and positive LandScan USA 2021
modeled daytime population; tract exposure is the sum of those BG-level maxima. LandScan excludes
tourists and other transitory visitors, so visitor-heavy areas remain flagged rather than fully repaired.
`raw_rate_*` fields are diagnostic direct count rates. `rate_*_primary` and
`index_*_primary` fields are the published count-derived primary rates and indexes. `rate_*_resident`
and `index_*_resident` fields are clearly labeled secondary resident-population series for every
offense and use the same count-derived estimator.
`diagnostic_eb_*` and `diagnostic_resident_eb_*` fields preserve the empirical-Bayes diagnostics
for audit purposes only.

## Index Formulas

For a geography `g`, offense `c`, with `count` the modeled annual count and `denom` the
offense-specific primary denominator:

Primary offense indexes:

```text
rate_g,c           = count_g,c / denom_g,c x 100,000
national_rate_c    = 100,000 x sum(count_g,c) / sum(denom_g,c)
index_g,c_primary  = 100 x rate_g,c / national_rate_c
```

The published rate/index path is count-derived and does not reference the empirical-Bayes posterior.
`raw_rate_*` equals the published `rate_*_primary` on published rows. `diagnostic_eb_*` fields carry
diagnostic posterior and shrinkage values only. Suppression is validity-only: rate and index fields
are null when publication eligibility or denominator validity fails. Denominators are never switched
row by row, high indexes are not capped or blanked, and infinite rates are not reported.

Secondary resident indexes:

```text
rate_g,c_resident        = count_g,c / resident_population_g x 100,000
resident_national_rate_c = 100,000 x sum(count_g,c) / sum(resident_population_g)
index_g,c_resident       = 100 x rate_g,c_resident / resident_national_rate_c
```

Aggregate indexes are explicit:

- `index_total_part1_resident`: AGS-comparable event-unweighted total Part-I resident index.
- `index_personal_part1_resident`: personal-offense resident index.
- `index_property_part1_resident`: property-offense resident index.
- `index_total_primary_event_weighted`: seven-offense primary-index composite weighted by national expected-count share.
- `index_total_equal_offense`: unweighted mean of the seven primary offense indexes.
- `index_total_harm`: Cambridge Crime Harm Index-style sentencing-days weighted primary-index composite.

Important interpretation fields:

- `primary_denominator_type_*`: offense-specific primary denominator family.
- `landscan_day_pop`: LandScan USA 2021 modeled daytime population used as a person-exposure
  denominator floor-lifter where larger than the jobs-based proxy.
- `landscan_day_lifted_person_exposure`: true when LandScan day population lifts the
  person-exposure denominator above the jobs-based proxy.
- `estimate_mode_*`: publication state for each offense, including count-derived rows and
  non-residential, special-use, insufficient-exposure, or vehicle-denominator-invalid suppression.
- `raw_rate_*`: diagnostic direct primary rate per 100,000 denominator units; equals the published
  `rate_*` on published rows.
- `diagnostic_eb_rate_*`, `diagnostic_eb_national_rate_per_100k_*`,
  `diagnostic_eb_prior_rate_*`, `diagnostic_eb_k_*`, `diagnostic_eb_observed_weight_*`, and
  `diagnostic_eb_prior_weight_*`: empirical-Bayes diagnostics, retained for audit only.
- `diagnostic_eb_low_denominator_flag_*`, `diagnostic_eb_heavy_shrinkage_flag_*`, and
  `diagnostic_eb_extreme_shrinkage_flag_*`: diagnostic quality flags.
- `primary_index_suppressed_*`: primary rate/index is suppressed because the primary denominator
  is at or below the structural hard minimum.
- `primary_zero_denominator_positive_count_*`: positive count was allocated to a zero primary
  denominator and should be inspected in the zero-target audit table.
- `index_*_resident_suppressed`: secondary resident rate/index is suppressed because resident
  population is at or below the structural hard minimum.
- `rate_*_resident`: published direct resident rate per 100,000 residents; equals
  `resident_raw_rate_*` on published rows and is null when suppressed.
- `index_*_resident`: published direct resident index normalized to national average 100.
- `diagnostic_resident_eb_rate_*`, `diagnostic_resident_eb_national_rate_per_100k_*`,
  `diagnostic_resident_eb_prior_rate_*`, `diagnostic_resident_eb_k_*`,
  `diagnostic_resident_eb_observed_weight_*`, and
  `diagnostic_resident_eb_prior_weight_*`: resident empirical-Bayes diagnostics, retained for
  audit only.
- `diagnostic_resident_eb_low_denominator_flag_*`,
  `diagnostic_resident_eb_heavy_shrinkage_flag_*`, and
  `diagnostic_resident_eb_extreme_shrinkage_flag_*`: resident diagnostic quality flags.
- `population_zero_with_positive_count`: the row has no resident population but nonzero modeled
  crime count.
- `crime_density_*`: expected incidents per square mile, emitted even where per-denominator
  rate/index fields are suppressed.
- `source_mode_*`, `feed_*`, `domain_overlap_score_*`, `confidence_tier_*`, and
  `confidence_reasons_*`: source-mode and confidence metadata appended after point estimates.

Column-level details are included in `materials/tables/output_schema.csv`.
"""
    (out_dir / "OUTPUT_SCHEMA.md").write_text(text)


def _write_repro_data_readme(repro_root: Path) -> None:
    data_dir = repro_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    text = """# Data Directory

Raw source data go here. The package intentionally ships without them.

Place each file at the exact path listed in `../../materials/tables/required_inputs.csv`
(acquisition directions are in that file's `how_to_obtain` column and in the package
README). Then check completeness with `uv run python main.py build-input-manifest` and
rebuild with `uv run python main.py build-release --emit-fbi-calibrated`.
"""
    (data_dir / "README.md").write_text(text)


def build_submission_package(*, repo_root: Path, out_dir: Path, force: bool) -> dict[str, object]:
    if out_dir.exists():
        if not force:
            raise SystemExit(f"Refusing to overwrite existing package directory: {out_dir}")
        _remove_existing_package(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repro_root = out_dir / "repro"
    repro_root.mkdir(parents=True, exist_ok=True)

    copied_paths: list[str] = []
    for rel_path in REPRO_RELATIVE_PATHS:
        src = repo_root / rel_path
        if not src.exists():
            raise SystemExit(f"Missing package path: {src}")
        _copy_path(src, repro_root / rel_path)
        copied_paths.append(f"repro/{rel_path}")

    copied_reference_paths: list[str] = []
    for src_rel, dst_rel in REFERENCE_COPY_PATHS.items():
        src = repo_root / src_rel
        if not src.exists():
            continue
        _copy_path(src, out_dir / dst_rel)
        copied_reference_paths.append(f"{src_rel} -> {dst_rel}")

    copied_validation_paths: list[str] = []
    for src_rel, dst_rel in VALIDATION_COPY_PATHS.items():
        src = repo_root / src_rel
        if not src.exists():
            continue
        _copy_path(src, out_dir / dst_rel)
        copied_validation_paths.append(f"{src_rel} -> {dst_rel}")

    analysis_summary = generate_submission_materials(
        repo_root=repo_root,
        out_dir=out_dir,
        package_repo_prefix="repro",
    )
    if MANUAL_INPUT_INVENTORY_MD.exists():
        _copy_path(MANUAL_INPUT_INVENTORY_MD, out_dir / "report" / "Manual_Input_Inventory.md")
    if MANUAL_INPUT_INVENTORY_CSV.exists():
        _copy_path(
            MANUAL_INPUT_INVENTORY_CSV,
            out_dir / "materials" / "tables" / "manual_input_inventory.csv",
        )
    _write_package_readme(out_dir)
    _write_repro_readme(repro_root)
    _write_output_schema_readme(out_dir)
    _write_repro_data_readme(repro_root)
    _prune_package_noise(out_dir)
    zip_path = _write_zip_archive(out_dir)

    return {
        "out_dir": str(out_dir),
        "zip_path": str(zip_path),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "repro_dir": str(repro_root),
        "copied_paths": copied_paths,
        "copied_reference_paths": copied_reference_paths,
        "copied_validation_paths": copied_validation_paths,
        "generated_report": analysis_summary["report_path"],
        "generated_materials_dir": analysis_summary["materials_dir"],
        "omitted_roots": [
            "archive/",
            ".venv/",
            "configs/templates/",
            "docs/TRACKER.md",
            "scripts/build/",
            "scripts/pull/",
            "scripts/review/",
            "scripts/diagnostics/benchmark_*",
            "scripts/diagnostics/feature_sanity.py",
            "data/ except promoted Overture feature artifacts",
            "state/cache/",
            "state/controls/ except validator support files",
            "state/geometry/",
            "state/locks/",
            "state/logs/",
            "state/modeling/non-promoted BG-prior variants",
            "state/modeling/city_incident_share_surface.parquet",
            "state/modeling/inputs/",
            "state/observations/ except validator support files",
            "state/reference/ except validator support files",
            "state/review/analysis/",
            "state/review/packets/source/",
            "state/review/runs/",
            "state/review/support/",
        ],
    }


def main() -> None:
    default_out = REPO_ROOT.parent / f"{REPO_ROOT.name}-submission-package"

    parser = argparse.ArgumentParser(
        description=(
            "Create the final submission package: a cleaned product-facing repo surface plus a "
            "detailed report and derived materials (tables, charts, and maps) from the canonical "
            "2024 build artifacts."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the package directory if it already exists.",
    )
    args = parser.parse_args()

    summary = build_submission_package(
        repo_root=REPO_ROOT,
        out_dir=args.out_dir.resolve(),
        force=bool(args.force),
    )
    print(summary)


if __name__ == "__main__":
    main()
