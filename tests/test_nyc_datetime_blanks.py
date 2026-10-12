import numpy as np
import pandas as pd

from crimerisk.city_incidents import _parse_nyc_datetime


def test_blank_fields_parse_the_same_from_csv_nan_and_parquet_none():
    rows = {
        "cmplnt_fr_dt": ["2018-01-01T00:00:00.000", "2018-01-02T00:00:00.000", "2017-12-29T00:00:00.000"],
        "cmplnt_fr_tm": ["17:22:00", "02:00:00", "12:01:00"],
        "cmplnt_to_dt": ["2018-01-01T00:00:00.000", np.nan, "2018-01-03T00:00:00.000"],
        "cmplnt_to_tm": [np.nan, "10:00:00", np.nan],
        "rpt_dt": ["2018-01-01T00:00:00.000", "2018-01-02T00:00:00.000", "2018-01-13T00:00:00.000"],
    }
    from_csv = pd.DataFrame(rows)
    from_parquet = from_csv.astype(object).where(from_csv.notna(), None)
    a = _parse_nyc_datetime(from_csv)
    b = _parse_nyc_datetime(from_parquet)
    pd.testing.assert_series_equal(a, b)
    # A missing end time falls back to the start time, not to midnight.
    assert a.iloc[0] == pd.Timestamp("2018-01-01 17:22:00")
