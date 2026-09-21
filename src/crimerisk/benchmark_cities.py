from __future__ import annotations


BENCHMARK_CITY_SPECS: tuple[tuple[str, str, str], ...] = (
    ("austin", "Austin", "48"),
    ("baltimore", "Baltimore", "24"),
    ("boston", "Boston", "25"),
    ("chicago", "Chicago", "17"),
    ("denver", "Denver", "08"),
    ("mesa", "Mesa", "04"),
    ("minneapolis", "Minneapolis", "27"),
    ("new_york", "New York", "36"),
    ("philadelphia", "Philadelphia", "42"),
    ("san_francisco", "San Francisco", "06"),
    ("seattle", "Seattle", "53"),
    ("washington_dc", "Washington", "11"),
)

BENCHMARK_CITY_KEYS: tuple[str, ...] = tuple(key for key, _name, _state_fips in BENCHMARK_CITY_SPECS)
BENCHMARK_CITY_KEY_SET: frozenset[str] = frozenset(BENCHMARK_CITY_KEYS)
BENCHMARK_CITY_NAMES: tuple[str, ...] = tuple(name for _key, name, _state_fips in BENCHMARK_CITY_SPECS)
BENCHMARK_CITY_STATE_FIPS: dict[str, str] = {
    key: state_fips for key, _name, state_fips in BENCHMARK_CITY_SPECS
}
