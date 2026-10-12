from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from sklearn.impute import SimpleImputer


MONOTONE_GAM_FEATURE_SPECS: tuple[tuple[str, str], ...] = (
    ("log_jobs_density_sqkm", "monotonic_inc"),
    ("log_daytime_population_jobs_proxy_density_sqkm", "monotonic_inc"),
    ("jobs_per_capita", "monotonic_inc"),
    ("tract_poverty_rate", "monotonic_inc"),
    ("snap_received_share", "monotonic_inc"),
    ("renter_units_per_capita", "monotonic_inc"),
    ("nlcd_developed_high_share", "monotonic_inc"),
    ("commute_transit_share", "monotonic_inc"),
)
MIN_MONOTONE_GAM_FEATURES = 4


def monotone_gam_feature_columns(feature_cols: list[str] | tuple[str, ...]) -> list[str]:
    available = {str(col) for col in feature_cols}
    selected = [
        str(column)
        for column, _constraint in MONOTONE_GAM_FEATURE_SPECS
        if str(column) in available
    ]
    if len(selected) < int(MIN_MONOTONE_GAM_FEATURES):
        raise RuntimeError(
            "Monotone GAM feature selection found too few usable columns "
            f"({len(selected)} < {MIN_MONOTONE_GAM_FEATURES})."
        )
    return selected


@dataclass
class MonotoneGAMRegressor:
    feature_names: list[str]
    lam: float = 0.6
    n_splines: int = 12
    max_iter: int = 200

    def __post_init__(self) -> None:
        self.feature_names = list(self.feature_names)
        self._imputer: SimpleImputer | None = None
        self._model: LinearGAM | None = None

    def _prepare_design(self, x: pd.DataFrame, *, fit: bool) -> np.ndarray:
        frame = pd.DataFrame(x).loc[:, self.feature_names].copy()
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if fit:
            self._imputer = SimpleImputer(strategy="median")
            return self._imputer.fit_transform(frame)
        if self._imputer is None:
            raise RuntimeError("MonotoneGAMRegressor must be fit before predict.")
        return self._imputer.transform(frame)

    def _build_gam(self) -> LinearGAM:
        spec_map = dict(MONOTONE_GAM_FEATURE_SPECS)
        terms = None
        for idx, column in enumerate(self.feature_names):
            term = s(
                idx,
                constraints=str(spec_map[str(column)]),
                n_splines=int(self.n_splines),
                lam=float(self.lam),
            )
            terms = term if terms is None else terms + term
        if terms is None:
            raise RuntimeError("MonotoneGAMRegressor requires at least one feature.")
        return LinearGAM(terms=terms, fit_intercept=True, max_iter=int(self.max_iter))

    def fit(self, x: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "MonotoneGAMRegressor":
        design = self._prepare_design(x, fit=True)
        self._model = self._build_gam()
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self._model.fit(design, np.asarray(y, dtype=float), weights=weights)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MonotoneGAMRegressor must be fit before predict.")
        design = self._prepare_design(x, fit=False)
        return np.asarray(self._model.predict(design), dtype=float)
