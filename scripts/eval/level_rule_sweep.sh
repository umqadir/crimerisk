#!/bin/bash
# Run the level-lane holdout once per rule configuration and append to one results table.
#
# Each configuration is scored on the arms the rule can reach: the admission rules are
# only visible on the corrupted-vector arm, the ladder rules only on the masked arms.
# The panel and the truth universe are cached and shared, so the sweep pays for the
# panel build once.
set -euo pipefail

cd "$(dirname "$0")/../.."

YEAR="${YEAR:-2024}"
OUT="${OUT:-state/eval/level_v1}"
EXTRA="${EXTRA:-}"

run() {
  local label="$1" rules="$2"; shift 2
  echo "=== $label [$rules] arms: $* ==="
  CRIMERISK_LEVEL_RULES="$rules" uv run python scripts/eval/level_holdout.py \
    --year "$YEAR" --label "$label" --out-dir "$OUT" --append --skip-remainder \
    --arms "$@" $EXTRA
}

# The three arms share one estimator pass over disjoint agency folds, so every
# configuration is one run and every configuration masks exactly the same agencies.
[ "${SKIP_BASELINE:-0}" = "1" ] || run baseline none refused silent corrupted
run rule_a_own_history_bound own_history_admission_bound refused silent corrupted
run rule_b_joint_vector joint_vector_gate refused silent corrupted
run rule_ab own_history_admission_bound,joint_vector_gate refused silent corrupted
run rule_c_nibrs_guard nibrs_transition_guard refused silent corrupted
run rule_d_own_history_pool prefer_own_history_over_pool refused silent corrupted
run all_rules all refused silent corrupted

# The state-remainder lane does not depend on the agency rules, and its input artifact
# can be rewritten by a controls build, so it is scored once, at the end, on whatever
# smoothed controls are current.
echo "=== state remainder leave-one-state-out ==="
CRIMERISK_LEVEL_RULES=none uv run python scripts/eval/level_holdout.py \
  --year "$YEAR" --label remainder_lane --out-dir "$OUT" --append --arms \
  --limit-agencies 1 || true

echo "sweep complete -> $OUT/results.csv"
