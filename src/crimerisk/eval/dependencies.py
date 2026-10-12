"""What an evaluation run consumes, read off the dependency declarations of the stages it runs.

Every build stage names its inputs for its own freshness check (`*_dependency_paths`). An
evaluation's input manifest is the union of those declarations over the stages the evaluation
runs or may rebuild, plus the files its own harness reads directly, plus the code and
configuration. Built this way it cannot omit an input a stage declares (the 2025.3 review found
the level holdout's hand-written list missing the CDE state-estimate file the smoother reads),
and the harness and the release validator call the same function, so the set a run records and
the set the validator requires of it cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# The code and configuration every evaluation is bound to.
CODE_AND_CONFIG_PATTERNS = (
    "main.py",
    "pyproject.toml",
    "uv.lock",
    "src/crimerisk/**/*.py",
    "configs/**/*",
)
LEVEL_HOLDOUT_SCRIPT = "scripts/eval/level_holdout.py"


def code_and_config_files(repo_root: Path, patterns: Iterable[str] = CODE_AND_CONFIG_PATTERNS) -> list[Path]:
    """Files matching `patterns` under `repo_root`, skipping caches and dotfiles."""
    root = Path(repo_root)
    return sorted(
        {
            path
            for pattern in patterns
            for path in root.glob(pattern)
            if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith(".")
        }
    )


def existing_files(paths: Iterable[Path | None]) -> list[Path]:
    """The declared paths that are files, once each (by resolved path), in declaration order.

    A declared input that is absent is not consumed (each reader either skips it or fails the
    stage), so it is not bound; when it appears it joins the set and unbinds every earlier run.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        key = path.resolve()
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        out.append(path)
    return out


def return_a_monthly_profile_paths(paths) -> list[Path]:
    """Every Return A master monthly file `observations.return_a_master_offense_months` reads.

    That reader takes every dated parse of every year (later parses win row by row), which is
    wider than the newest-parse-per-year set `observations.return_a_master_dependency_paths`
    declares; the masked-gap detector of every fold calls it directly.
    """
    from crimerisk.observations import RETURN_A_MASTER_PARSED_GLOB

    data_dir = Path(paths.data_dir)
    return [
        path
        for directory in sorted(data_dir.glob(RETURN_A_MASTER_PARSED_GLOB.format(year="*")))
        for path in sorted(directory.glob("return_a_master_monthly_*.parquet"))
    ]


def _controls_closure(paths, *, year: int, panel_year_end: int) -> list[Path]:
    """The declared inputs of the agency estimator, the controls stage and the smoother, and of
    the reporting-regime and observation stages a stale panel build refreshes, plus every Return
    A monthly profile the estimator's masked-gap detector reads (`return_a_monthly_profile_paths`;
    no stage declares the full set)."""
    from crimerisk.allocation import _agency_estimates_dependency_paths
    from crimerisk.controls import _controls_dependency_paths
    from crimerisk.level_lane import admission_dependency_paths
    from crimerisk.reporting_regimes import (
        ReportingRegimeBuildConfig,
        reporting_regime_dependency_paths,
    )
    from crimerisk.service_scopes import service_wide_scope_dependency_paths
    from crimerisk.smoothed_controls import smoothed_controls_dependency_paths

    return [
        *_agency_estimates_dependency_paths(paths, year=int(year)),
        *_controls_dependency_paths(paths, year=int(year)),
        *smoothed_controls_dependency_paths(paths, year=int(year)),
        *reporting_regime_dependency_paths(
            paths,
            config=ReportingRegimeBuildConfig(
                year_start=2018, year_end=max(int(year), int(panel_year_end))
            ),
        ),
        *admission_dependency_paths(paths),
        *service_wide_scope_dependency_paths(paths),
        *return_a_monthly_profile_paths(paths),
    ]


def level_holdout_dependency_paths(
    paths, *, year: int, panel_year_end: int, remainder: bool
) -> list[Path]:
    """Every file a level-holdout run consumes (`scripts/eval/level_holdout.py`).

    The fold build runs the panel build, admission, the agency estimator, the controls stage and
    the smoother, so it consumes their declared inputs (`_controls_closure`), among them the CDE
    state-estimate snapshot for `year`, the agency and block-group crosswalks, the reference
    tables and the raw sources behind the observations and reporting regimes. Inputs a stage
    declares under state/controls are dropped: the fold builds those itself on an empty, isolated
    state/controls (`level_holdout.isolated_state_paths`) and never reads production's. Added are
    what the harness reads directly: the observations (anchor inputs), the crosswalks of the
    dropped-agency fallback, the production
    smoothed controls when the state-remainder lane is scored, the holdout script, and the code
    and configuration (`CODE_AND_CONFIG_PATTERNS`).
    """
    from crimerisk.smoothed_controls import smoothed_controls_path

    root = Path(paths.repo_root)
    state = Path(paths.state_dir)
    fold_built = (state / "controls").resolve()
    declared = [
        path
        for path in _controls_closure(paths, year=year, panel_year_end=panel_year_end)
        if path is not None and not Path(path).resolve().is_relative_to(fold_built)
    ]
    direct = [
        state / "observations" / "agency_year_observations.parquet",
        state / "reference" / "agency_to_jurisdiction_crosswalk.parquet",
        state / "geometry" / "block_group_to_jurisdiction_crosswalk.parquet",
        *([smoothed_controls_path(paths, year=int(year))] if remainder else []),
        root / LEVEL_HOLDOUT_SCRIPT,
    ]
    return existing_files([*declared, *direct, *code_and_config_files(root)])


def build_outputs_dependency_paths(paths, *, year: int) -> list[Path]:
    """The declared inputs of every stage a `main.py build-outputs` fold build runs or refreshes.

    Observations, reporting regimes, the agency estimator (with every Return A monthly profile),
    controls and smoothing (`_controls_closure`); reference layers; geometry (enumerated per state from the jurisdiction
    master, when there is one); the block-group features and stage-10b prior; the exposure
    normalizers and mixture experts; and the city incident shares with every raw feed file. The
    raw inputs under data/ are among them, so a fold that would rebuild a derived input from a
    changed raw file is bound to that file.
    """
    from crimerisk.city_shares import _city_dependency_paths, _city_impls
    from crimerisk.exposure_ensemble import ExposureEnsembleConfig, exposure_normalizer_dependency_paths
    from crimerisk.geometry import geometry_dependency_paths
    from crimerisk.mixture_allocation import MixtureAllocationConfig, mixture_experts_dependency_paths
    from crimerisk.model_surface import ModelSurfaceConfig, model_prior_dependency_paths
    from crimerisk.pa_police_service import pa_police_service_dependency_paths
    from crimerisk.reference_layers import reference_layer_dependency_paths

    year = int(year)
    jurisdiction_master = Path(paths.state_dir) / "reference" / "jurisdiction_master.parquet"
    return existing_files(
        [
            *_controls_closure(paths, year=year, panel_year_end=year),
            *reference_layer_dependency_paths(paths=paths),
            *(geometry_dependency_paths(paths) if jurisdiction_master.is_file() else []),
            *model_prior_dependency_paths(paths, config=ModelSurfaceConfig(year=year)),
            *exposure_normalizer_dependency_paths(paths, config=ExposureEnsembleConfig(year=year)),
            *mixture_experts_dependency_paths(paths, config=MixtureAllocationConfig(year=year)),
            *_city_dependency_paths(paths, _city_impls(paths)),
            *pa_police_service_dependency_paths(paths),
        ]
    )
