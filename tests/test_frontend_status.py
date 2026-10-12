import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BUILD = Path(__file__).parents[1] / "frontend" / "build"
sys.path.insert(0, str(BUILD))
SPEC = importlib.util.spec_from_file_location("frontend_extract", BUILD / "01_extract.py")
EXTRACT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXTRACT)

import crschema  # noqa: E402

PUB = crschema.STATUS_PUBLISHED
SUP = crschema.STATUS_SUPPRESSED
TRACT = crschema.STATUS_TRACT_ONLY
NONE = crschema.STATUS_NO_DATA


def surface(n: int) -> pd.DataFrame:
    """All 20 measures published, no suppression flag set."""
    df = pd.DataFrame(index=range(n))
    for o in crschema.OFFENSES:
        short = crschema.SHORT[o]
        df[f"x_{short}"] = 100.0
        df[f"r_{short}"] = 100.0
        df[f"primary_index_suppressed_{o}"] = False
        df[f"index_{o}_resident_suppressed"] = False
    for key in crschema.COMPOSITES:
        df[f"x_{key}"] = 100.0
        df[f"r_{key}"] = 100.0
    return df


def test_offense_status_reads_the_suppression_flag_and_the_support_policy():
    df = surface(3)
    # row 0 published; row 1 null and flagged; row 2 null and not flagged
    df.loc[1:, "x_rob"] = np.nan
    df.loc[1, "primary_index_suppressed_robbery"] = True
    df.loc[1:, "r_bur"] = np.nan
    df.loc[1, "index_burglary_resident_suppressed"] = True

    tract = EXTRACT.offense_status(df, "tract")
    assert tract["x_rob"].tolist() == [PUB, SUP, NONE]
    assert tract["r_bur"].tolist() == [PUB, SUP, NONE]
    assert tract["x_mur"].tolist() == [PUB, PUB, PUB]

    bg = EXTRACT.offense_status(df, "bg")
    # Murder and rape at block group are tract-only whatever their own value or flag.
    for key in ("x_mur", "r_mur", "x_rap", "r_rap"):
        assert bg[key].tolist() == [TRACT, TRACT, TRACT]
    assert bg["x_rob"].tolist() == [PUB, SUP, NONE]


def test_block_group_exposure_composite_is_gated_by_the_parent_tract_murder_term():
    df = surface(2)
    df["x_ov"] = [np.nan, 100.0]
    df["x_vi"] = [np.nan, 100.0]
    df["x_pr"] = [np.nan, 100.0]
    off = EXTRACT.offense_status(df, "bg")
    parent = {
        "x_mur": np.array([SUP, PUB], dtype="int8"),
        "x_rap": np.array([PUB, PUB], dtype="int8"),
    }
    comp = EXTRACT.composite_status(df, "bg", off, parent)
    assert comp["x_ov"].tolist() == [SUP, PUB]
    assert comp["x_vi"].tolist() == [SUP, PUB]
    # Property has no murder term, so a null property composite with every gate
    # published is "no data", not a suppression.
    assert comp["x_pr"].tolist() == [NONE, PUB]


def test_block_group_resident_composite_ignores_murder_and_rape():
    df = surface(2)
    df["r_ov"] = [np.nan, np.nan]
    df.loc[1, "r_lar"] = np.nan
    df.loc[1, "index_larceny_resident_suppressed"] = True
    off = EXTRACT.offense_status(df, "bg")
    parent = {"x_mur": np.array([SUP, SUP]), "x_rap": np.array([SUP, SUP])}
    comp = EXTRACT.composite_status(df, "bg", off, parent)
    # Row 0: the only suppressed thing is the parent tract's murder, which does not
    # gate the resident composite, so the null composite is not explained -> no data.
    assert comp["r_ov"].tolist() == [NONE, SUP]


def test_block_group_composite_status_requires_the_parent():
    df = surface(1)
    off = EXTRACT.offense_status(df, "bg")
    with pytest.raises(ValueError):
        EXTRACT.composite_status(df, "bg", off, None)


def test_packed_status_round_trips_every_measure_and_stays_exact_in_js():
    rng = np.random.default_rng(7)
    status = {m: rng.integers(0, 4, size=50).astype("int8") for m in crschema.MEASURE_ORDER}
    packed = EXTRACT.pack_status(status)
    for i in range(50):
        for m in crschema.MEASURE_ORDER:
            assert EXTRACT.unpack_status(int(packed[i]), m) == status[m][i]
    worst = EXTRACT.pack_status({m: np.array([3]) for m in crschema.MEASURE_ORDER})
    assert int(worst[0]) < 2 ** 53
    # The client reads a digit with floor(ms / 4**i) % 4 in float arithmetic.
    for i, m in enumerate(crschema.MEASURE_ORDER):
        assert int((float(worst[0]) // float(4 ** i)) % 4) == 3


def test_suppression_floors_are_read_from_the_parquet(tmp_path):
    path = tmp_path / "surface.parquet"
    pd.DataFrame(
        {"person_exposure_denominator_floor": [50.0, 50.0], "other": [1, 2]}
    ).to_parquet(path)
    assert EXTRACT.suppression_floors(path) == {"person_exposure_denominator_floor": 50.0}

    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"person_exposure_denominator_floor": [50.0, 75.0]}).to_parquet(bad)
    with pytest.raises(ValueError):
        EXTRACT.suppression_floors(bad)


def test_the_resident_card_quotes_the_per_resident_floor_not_the_exposure_floor():
    from crimerisk.allocation import RESIDENT_RATE_MIN_RESIDENTS, RESIDENT_RATE_MIN_RESIDENTS_ORDINARY

    floors = {
        "person_exposure_denominator_floor": 50.0,
        "resident_rate_population_floor": 550.0,
        "resident_rate_population_floor_ordinary": 300.0,
    }
    assert RESIDENT_RATE_MIN_RESIDENTS == floors["resident_rate_population_floor"]
    assert RESIDENT_RATE_MIN_RESIDENTS_ORDINARY == floors["resident_rate_population_floor_ordinary"]
    assert crschema.STATUS_CARD["suppressed_resident"].format(**floors) == (
        "Not published: fewer than 300 residents (550 in a special-use area)."
    )
