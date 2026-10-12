"""The published surface's geographic scope, defined once.

The product is a CONUS surface: the 48 contiguous states plus the District of
Columbia. Two populations sit outside it and are excluded by name rather than by
each consumer remembering to filter:

* **Non-contiguous states** (AK, HI) -- excluded from the geometry, covariate and
  control builds, so an agency estimate for them has nothing to land on.
* **UCR territory and non-state codes** (AS, CZ, GM, GU, MP, PR, VI) -- Puerto
  Rico, Guam, the Virgin Islands, the Northern Marianas, American Samoa and the
  retired Canal Zone. They are real FBI submissions (PR alone carries 118k counts
  across 2018-2024) but they are not in the census geography the surface is built
  on, so they can never be allocated.

`NB` is deliberately NOT here: those two ORIs (federal ATF/FBI Omaha offices) sit
in Nebraska under the retired two-letter state code and carry zero counts; they
are handled by the dead-ORI predicate, not by a scope rule. `GM` IS here -- it is
a Guam ATF ORI carrying state FIPS 66.

Four modules used to keep their own copy of the exclusion set; they now import
this one. `fbi_publications` keeps a separate, narrower set for CIUS name
matching -- that is a parsing-time filter over published tables, not the product
scope, and it is not this constant.
"""

from __future__ import annotations

import pandas as pd


NON_CONTIGUOUS_STATE_ABBRS = frozenset({"AK", "HI"})
TERRITORY_STATE_ABBRS = frozenset({"AS", "CZ", "GM", "GU", "MP", "PR", "VI"})
PRODUCTION_SCOPE_EXCLUDE = frozenset(NON_CONTIGUOUS_STATE_ABBRS | TERRITORY_STATE_ABBRS)


def production_scope_excluded(state_abbr: pd.Series) -> pd.Series:
    """True where the row's state code is outside the published CONUS scope."""
    return (
        state_abbr.astype("string").str.upper().isin(PRODUCTION_SCOPE_EXCLUDE).fillna(False)
    )
