# CrimeRisk — Pipeline Contracts

A description of the pipeline as it exists, one section per stage: what the stage takes,
what it emits, what it enforces, and every manual override or registry that touches it,
in application order. **This document records what the code does today.** Where the
behaviour differs from what the surrounding comments and docs claim, the actual behaviour
is written down and marked `SURPRISE:`.

---

# Stage 1 — FBI record ingestion → the agency-year observations panel

Scope: everything from the raw FBI/state/local crime records up to and including the
canonical per-agency-year-offense panel and the preferred-observation selection over it.
It ends at `build_agency_preferred_observations()`. Agency→footprint resolution
(crosswalk, jurisdiction assignment) is Stage 2 and is out of scope here even where the
same files are read.

Entry points:

| Artifact | Builder | Path |
|---|---|---|
| agency-year observations panel | `crimerisk.observations.build_agency_year_observations` | `state/observations/agency_year_observations.parquet` |
| jurisdiction-year rollup of the same | `crimerisk.observations.build_jurisdiction_year_observations` | `state/observations/jurisdiction_year_observations.parquet` |
| reporting regimes | `crimerisk.reporting_regimes.build_agency_year_reporting_regimes` | `state/modeling/agency_year_reporting_regimes.parquet` |
| preferred observations (per target year) | `crimerisk.source_selection.build_agency_preferred_observations` | in-memory; materialised by `trend_fills.build_agency_trend_fill_panel` |

Configured span: `ObservationBuildConfig(year_start=2018, year_end=2024)`.
Offense vocabulary: the 7 Part I offenses in `crimerisk.crime.OFFENSES_7` —
`murder, rape, robbery, aggravated_assault, burglary, larceny, motor_vehicle_theft`.

## 1.1 Raw inputs and their lanes

Five source lanes feed the panel. Each lane emits rows carrying `source`,
`source_family`, `source_origin`, `raw_data_source`, `source_lane`, `reporting_mode`,
`conversion_status` (labels defined in `src/crimerisk/source_provenance.py`).

**Lane A — `srs_return_a_annual` (SRS / Return A).**
Files: `data/SRS-Kaplan-1960-2024/offenses_known_parquet_1960_2024_year.zip` (annual) and
`…_month.zip` (monthly, one cached parquet per year). Built by
`observations._build_srs_annual_observations`. ORI9 is `COALESCE(NULLIF(ori9,''), ori||'00')`.
Counts come from `crime.srs.SRS_OFFENSE_COLUMN_MAP`. Monthly file supplies
`monthly_row_count`, `non_missing_month_count`, `nonzero_months`, `max_month_total`, which
drive `max_month_share`, `annual_month_diff_ratio` and `monthly_lumpiness_flag`. Note that
post-2020 the majority of Return A rows are FBI-converted NIBRS, not native SRS
submissions — hence the label `as_released_return_a`, documented in
`source_provenance` and `docs/FBI-DATA-GUIDE.md`.

**Lane B — `nibrs_srs_equivalent_annual` (this repo's own NIBRS rollup).**
Files: `data/NIBRS-Kaplan-1991-2024/{offense,victim,property,batch_header}_segment_parquet_1991_2024.zip`.
Rolled up per year by `crime.nibrs.aggregate_nibrs_year_srs_equivalent` using the FBI's
documented SRS scoring rules (per-victim person crimes, rape = 11A+11B+11C, per-vehicle
MVT, hotel rule). Reported months come from the NIBRS **batch header**
(`number_of_months_reported`); where that is null or ≤0 it falls back to
`incident_months_any` — months in which the agency filed any incident.

**Lane C — `cius_publication_annual` (FBI CIUS annual tables).**
Files: `data/FBI-CIUS-Annual/<year>/raw/*` (xls/xlsx, resolved by
`fbi_publications.CIUS_TABLE_SPECS`). Three table specs are parsed —
`table8_city`, `table9_university`, `table11_state_tribal_other` — but only **table 8 and
table 9** are promoted into the panel (`observations._build_cius_summary_promotions`).
Table 11 is parsed and unused at this stage.

**Lane D — `state_publication_annual`.**
Canonical input surface `state/modeling/inputs/state_publication_annual.parquet`, built
by `crimerisk.state_publications` from three state programs:
`ny_dcjs_index_crimes_annual` (NY DCJS 2021-2024, `data/NY-DCJS-2024/parsed/`),
`fdle_fibrs_offense_detail_ytd` (FL FDLE FIBRS 2024, `data/FDLE-FIBRS-2024/parsed/`),
`ms_tops_offense_detail_annual` (MS TOPS 2024, `data/MS-TOPS-2024/parsed/`).
Current content: 16,975 rows — NY 3,675–3,724/yr for 2021-24, FL 1,421 (2024), MS 728 (2024).

**Lane E — `local_publication_annual`.**
Canonical input surface `state/modeling/inputs/local_publication_annual.parquet`, built
by `crimerisk.local_publications` from reviewed per-city packets. Current content: 61
rows / 6 cities (Atlanta GA, Brunswick GA, Calumet City IL, Quincy IL, Hallandale Beach
FL, New Orleans LA).

**Reference input — `state/reference/agency_master.parquet`** (built by
`reference_layers` / `reference.py`; Stage 2 owns it). Supplies `agency_name_std`,
`agency_type_norm`, `manual_review_flag`, `population_latest_nibrs`, and back-fills
`state_fips / state_abbr / county_fips / place_fips` where the crime file left them null.
`state/reference/jurisdiction_master.parquet` and
`state/reference/agency_to_jurisdiction_crosswalk.parquet` are read at this stage only to
disambiguate CIUS municipal name matches and to build the jurisdiction rollup.

## 1.2 What the panel is

`agency_year_observations.parquet` — currently **2,056,166 rows over 26,103 ORIs and
7 years (2018-2024)**, 44 columns.

**A row is one (`ori9`, `year`, `source`, `offense`) cell.** The primary key is exactly
that 4-tuple and it is unique (verified: 0 duplicate keys). A row is a *candidate*
observation, not a chosen one: the same agency-year-offense may appear in up to five
lanes and the panel keeps all of them side by side. Choosing among them is
`source_selection`'s job (§1.5).

Column groups:

* identity — `ori9, ori7, year, source, offense`
* provenance — `source_family, source_origin, raw_data_source, source_lane,
  reporting_mode, conversion_status, state_exception_flag, cius_reference_flag`
* value — `count`
* participation — `months_reported, months_missing, quality_tier, observation_weight,
  monthly_lumpiness_flag`, plus the SRS-only monthly diagnostics
  (`monthly_part1_total, monthly_row_count, nonzero_months, max_month_share,
  annual_month_diff_ratio, annual_part1_total`)
* annual-batch detector — `reported_months_original, annual_batch_detected,
  annual_batch_detector_reason, annual_batch_panel_median_full_year_total,
  annual_batch_panel_max_full_year_total, annual_batch_absolute_total_flag,
  annual_batch_panel_median_flag`
* geography/agency attributes carried from the crime file and agency_master —
  `state_fips, state_abbr, county_fips, place_fips, population, agency_name_raw,
  agency_name_std, agency_type_raw, agency_type_norm, crosswalk_agency_name,
  census_name, manual_review_flag`

`quality_tier` is a pure function of `months_reported`
(`_quality_tier_from_months`: 1-5 sparse, 6-9 low, 10-11 medium, ≥12 high, else unknown),
demoted one rung on the SRS lane when `monthly_lumpiness_flag` fires.
`observation_weight` is a fixed map off the tier (`high 1.0, medium 0.8, low 0.5,
sparse 0.25, unknown 0.1`).

### What `months_reported` means, per lane

| Lane | months_reported | Source |
|---|---|---|
| `srs_return_a_annual` | count of months with `month_missing = 0` in the SRS monthly file; falls back to `12 − number_of_months_missing` from the annual file when the monthly file has no rows | measured |
| `nibrs_srs_equivalent_annual` | NIBRS batch-header `number_of_months_reported`; falls back to `incident_months_any` when null/≤0 | measured (agency-year property, identical for all 7 offenses) |
| `cius_publication_annual` | **hard-set to 12.0** | asserted |
| `local_publication_annual` | **hard-set to 12.0** | asserted |
| `state_publication_annual` | source `months_reported` if present, else `quarters_reported × 3`, else **12.0**; clipped to [1,12] | mixed (NY/FL supply it, MS does not) |

**MS TOPS months — accepted with reason.** The MS TOPS extract is pulled from the
program's own report exports (`data/MS-TOPS-2024/raw/report_105` violent-crime trend and
`report_115` property-crime five-year change, one CSV per ORI). Those exports carry a
title block, a jurisdiction line, a `Measures: Number of Crimes` line, and then an
offense × calendar-year matrix. There is **no months-reported, quarters-reported or
coverage field anywhere in the source** — the only participation signal is a trailing
`Column suppression applied.` note, which marks suppressed cells rather than partial
years. The 728 rows / 133 ORIs / 32,222 counts therefore stay at the 12.0 default: it is
the program's own annual figure and no better coverage statement exists to carry. NY
DCJS and FL FDLE both publish a period and are read from it.

`months_missing` is always `12 − months_reported` (verified: 0 rows violate it).

### What zero-vs-missing means, per lane (post-v19)

* **SRS/Return A** — the lane has always emitted an explicit row per Part I offense, so
  a `0` is a reported zero over `months_reported` months and an absent ORI-year is
  missing. `months_reported = 0` with `count = 0` is a non-report and the regime ladder
  correctly routes it to `structurally_missing_or_unreliable`.
* **NIBRS rollup** — `_complete_nibrs_offense_rows_with_zeros` completes every
  **submitted** agency-year to all 7 offenses with explicit `count = 0`, because
  reporting in NIBRS is an agency-year property. Zeros are zeros over the batch header's
  months. The submitted population is the **batch header's** agency-years with
  `number_of_months_reported > 0`, unioned with any agency-year the offense rollup
  carries without a header record. Keying completion on the rollup alone (v19) lost the
  agency-years whose whole submission held no Part I incident — 8,306 agency-years over
  2018-2024, 1,631 in 2024, including the Los Angeles County and San Bernardino County
  sheriffs — and read them downstream as missing rather than as the all-zero years the
  header says they are.
* **CIUS** — a table-8/9 row is a completed annual compilation, so every promoted row is
  a full-year value. All-zero promoted agency-years therefore publish as observed
  full-year zeros at weight 1.0 (1,211 across the span, 314 in 2024; 897 of them have
  positive own history).

* **State publication** — `_drop_state_publication_non_reports` resolves the sheet's
  ambiguity: an agency-year that is zero for every offense in the state sheet **and**
  shows zero SRS months (or is absent from the SRS panel) is dropped as a non-report.
  Survivors publish as full-year zeros (19 across the span, 9 in 2024).
* **Local publication** — no zero/absence discrimination at all; every row published is
  a full-year value at months = 12.

### Offense-set completeness, per lane

The two FBI lanes claim a complete offense set per agency-year and are asserted to have
one (`_assert_fbi_lane_offense_sets_are_complete`, fail-closed): Return A emits an
explicit row per Part I offense, and the NIBRS rollup is completed as above. **The
publication lanes do not, and an absent offense there is MISSING, not zero** — 794
agency-years (CIUS 700, state publication 92, local 2) carry fewer than seven rows:

* **CIUS** withholds individual cells, with the reason printed in the table's own
  footnotes: *"The FBI determined that the agency's data were overreported.
  Consequently, those data are not included in this table."* (Table 8, e.g. Arkansas
  2018: Benton, Magnolia, Siloam Springs and El Dorado are footnoted and carry blank
  property-crime and burglary cells while their other columns publish normally.) The
  parser drops the blank cells, so the panel simply has no row for that (agency, year,
  offense) and selection falls to the next lane for it — the correct treatment of a
  withheld figure.
* **MS TOPS** prints `Column suppression applied.` on suppressed exports; those cells are
  likewise absent rather than zero.
* **Local publication** — two reviewed packets do not report aggravated assault.

A publication lane's missing offense inside an otherwise-chosen agency-year is what
§1.5's supplement rule handles.

## 1.3 Interventions, in the order they apply

Everything below is applied inside `build_agency_year_observations` in this order:

1. **`configs/agency_master_supplement.csv`** (100 rows) — applied upstream in
   `reference_layers`/`reference.py`, before Stage 1 reads `agency_master.parquet`. Adds
   ORIs (mostly CHP area offices and other NIBRS-only agencies) the SRS/NIBRS files alone
   would not resolve.
2. **County-FIPS canonicalisation** (`reference.canonicalize_agency_county_fips`, also
   upstream, on `agency_master`) — clears the `999` sentinel, remaps retired GEOIDs
   (`RETIRED_COUNTY_GEOID_REMAP`: 02261→02063, 02270→02158, 51515→51019, 46113→46102),
   then fills nulls from the FBI CDE agency roster and LEAIC, recording
   `county_fips_source`. Only `{srs_agency_header, retired_county_remap}` are
   county-anchor eligible (`COUNTY_ANCHOR_ELIGIBLE_COUNTY_FIPS_SOURCES`).
   The **same function** is applied to the observations panel at step 12 below, so the
   panel and the master share one implementation and one `county_fips_source`
   vocabulary; the panel's own `999` sentinels (98 rows) and retired GEOIDs (525 rows:
   02261, 02270, 46113, 51515, 57999) are gone.
3. **SRS lane build** → lumpiness flags, tiers, weights; merge `agency_master`.
4. **CIUS promotion** (`_build_cius_summary_promotions`) — table-8 municipal rows are
   matched to jurisdictions via `fbi_publications.match_cius_municipal_rows_to_jurisdictions`,
   kept only where the published name matches exactly one jurisdiction-offense **and**
   the jurisdiction has exactly one crosswalk ORI at weight > 0.999; table-9 rows are
   matched by agency alias to a unique `special_jurisdiction` ORI. Promotion is an
   **inner join onto that year's SRS rows**, so a CIUS row for an ORI with no SRS row that
   year is silently dropped. Promoted rows are forced to months 12 / tier high / weight 1.0.
5. **State-publication promotion** — load span 2018-2024, then
   `_drop_state_publication_non_reports` (see §1.2), then months/quarters resolution,
   tiers, weights. `state_exception_flag = True` on every surviving row.
6. **Local-publication promotion** — months forced to 12.
7. **Concat + `_deduplicate_agency_source_candidates`** — within a
   (`ori9, year, source, offense`) key, keep the row with the highest
   `observation_weight`, then `months_reported`, then `count`. This is what makes the
   primary key unique.
8. **NIBRS lane build** (offense completion → batch header join → months → tiers) and
   concat.
9. **Cross-lane twin identity resolution** (`agency_identity.build_cross_lane_twin_ledger`
    → `apply_cross_lane_twin_ledger`, then the dedupe rule again) — each NIBRS-lane ORI9
    variant (`…0X`, `…9E`, `…5Y`, …) whose seven-offense vector matches its `…00`
    summary-lane stem-mate exactly in a year with positive counts is re-keyed onto that
    `…00` ORI. Single-year single-offense agreements additionally require the FBI CDE
    roster to list the NIBRS ORI and not the summary one. **173 variants** resolve;
    resolution must be a function from variant to canonical or the build fails.
10. **Negative-count clamp** (`_clamp_return_a_negative_counts`) — Return A's
    unfounded-offense adjustment residues (29 rows, −37 counts, all `srs_return_a_annual`)
    are clamped to zero with the amount kept in `negative_count_clamped_amount`. A
    negative on any other lane fails the build.
11. **`_apply_annual_batch_detector`** — source-agnostic: a row with
    `months_reported ≤ 2`, a positive annual total, and either an annual total ≥ 500
    (`srs_lumpy_min_total`) or ≥ 60 % of the agency's 2019-2023 full-year panel median, is
    promoted to `months_reported = 12`, tier high, weight 1.0,
    `conversion_status = batched_annual`. `reported_months_original` preserves the
    pre-promotion value.
12. **County-FIPS canonicalisation of the panel** (`canonicalize_agency_county_fips`, the
    same function as step 2) and the **production-scope flag**
    (`scope.production_scope_excluded` → `production_scope_excluded`, true for
    `AK, HI, AS, CZ, GM, GU, MP, PR, VI`).
13. **Schema normalisation, provenance labelling, `state_fips` non-null filter**,
    fail-closed post-conditions (`_assert_no_negative_counts`,
    `_assert_fbi_lane_offense_sets_are_complete`), column ordering, sort by
    (`ori9, year, source, offense`).

Registries that touch the *selection* over the panel, applied inside
`reporting_regimes.build_agency_year_reporting_regimes` **after** the panel is written:

14. **`configs/reporting_regime_overrides.csv`** (194 rows) — keyed
    (`ori`,`year`,`offense`), forces `reporting_regime` to one of the five valid values;
    validated for membership and duplicates, fails closed otherwise. Sets
    `override_applied` and re-derives `preferred_source_by_regime` from lane presence.
15. **`configs/source_preference_overrides.csv`** (3 rows: Tacoma WA larceny, Louisville
    KY larceny) — keyed (`ori`,`year`,`offense`), forces `preferred_source_by_regime`
    and sets `source_override_applied`. Selection reads the lane it names as the choice
    for the whole agency-year (§1.5).

No other registry modifies Stage 1. `configs/consolidated_agency_footprints.csv`,
`local_resolution_overrides.csv`, `overlap_footprint_overrides.csv`,
`overlap_custom_footprints.csv`, `municipal_geometry_overrides.csv`,
`consolidated_agency_detector_exceptions.csv`, `state_police_post_county_footprints.csv`
and `tribal_agency_aiannh_footprints.csv` are Stage 2/3 registries and are named here only
so the boundary is explicit.

Stage 1 does contribute one Stage 2 *identity* column: `agency_master.is_tribal_agency`,
set by `reference.tribal_agency_flag` from the FBI CDE roster's `agency_type_name ==
"Tribal"` union (LEAIC `LG_POPULATION = 999999999` **and** a word-boundary tribal name
token). The sentinel is not used alone because LEAIC also applies it to police-protection
and community-services districts. That flag is what gates the automatic agency-seat-place
shortcut in Stage 2 (`assert_tribal_agencies_not_auto_placed`).

## 1.4 Reporting regimes

`agency_year_reporting_regimes.parquet` assigns one of five regimes per
(`ori9`,`year`,`offense`), by a first-match ladder (later rungs overwrite earlier ones in
source order — CIUS, local, state, SRS-batch, SRS-full, SRS-partial, SRS-lumpy, then the
two "unassigned" fallbacks):

| regime | meaning | downstream effect |
|---|---|---|
| `full_monthly` | SRS ≥12 months, not lumpy, monthly mask matches metadata | usable as observed |
| `true_partial` | 1-11 months on **any** lane with a coherent month mask | licenses ×12/months annualisation |
| `lumpy_or_batched` | lumpiness signal, or 1-11 months whose month mask contradicts the metadata | usable as observed |
| `annual_only_but_usable` | CIUS / complete publication / SRS annual batch / SRS annual without monthly support / **NIBRS with ≥12 months** | usable as observed |
| `structurally_missing_or_unreliable` | no positive and no month-bearing support on any lane | not usable; routes to the fill ladder |

The NIBRS rung keys on **months reported, not positive count** — that is what makes a
NIBRS true zero an observation rather than an absence — and it splits on coverage: a
batch header reporting 1-11 months is `true_partial` (`nibrs_batch_header_partial_year`),
12 or more is `annual_only_but_usable`. Partiality is a property of months, not of lane
family; before this split the NIBRS lane could not be partial at all and 8,006 rows over
1,410 agencies in 2024 were read as complete years at their partial value.

## 1.5 Preferred-observation selection

`source_selection.build_agency_preferred_observations(paths, year)`:

1. Load the whole panel; compute `_globally_dead_observation_oris` — ORIs whose total
   count is ≤0 **and** whose max months is ≤0 across *all* years — and exclude them
   entirely.
2. Restrict to the target year and the five known sources; pivot each lane to its own
   count / weight / months / conversion-status columns keyed
   (`ori9, state_fips, state_abbr, offense`); outer-join the five.
3. Join the reporting regimes for that year and the published-NIBRS reference counts
   (`published_nibrs.load_published_nibrs_reference_counts`, from
   `data/FBI-NIBRS-Tables-2024`).
4. **The lane is chosen once per (`ori9`, year), not per offense.** Every present lane is
   ranked within the agency-year by, in order: a manual source override naming it;
   publication-lane standing (CIUS, local, state — a completed annual compilation
   outranks either federal file's rendering of the same year); greater month coverage;
   greater observation weight; the FBI's published NIBRS tables corroborating the rollup
   where Return A disagrees; and finally the standing lane order
   **CIUS > local > state > SRS > NIBRS**. Coverage and weight are agency-year properties
   on every lane, so they aggregate as the maximum over the agency's offenses.
   `preferred_lane_for_agency_year` and `preferred_lane_selection_reason` record the
   outcome per row.
5. **Supplement, the one annotated exception.** Every offense reads the chosen lane. Where
   the chosen lane does not publish an offense at all (only the publication lanes ever
   do — see §1.2's completeness table), that offense alone falls to the next lane in the
   same agency-year ranking and is marked `preferred_source_is_lane_supplement`.
   `_assert_preferred_lane_unity` fails the build on any other split.
6. `preferred_count`, `preferred_observation_weight`, `preferred_months_reported`,
   `preferred_conversion_status` are copied from the selected lane's columns.
7. `trend_fills.add_preferred_support_flags` then derives the two flags every downstream
   consumer keys on, from the SELECTED lane's own months:
   * **supported** — the chosen lane reported the year at all, i.e. the regime is not
     `structurally_missing_or_unreliable`.
   * `current_row_is_true_partial` — supported, the chosen lane covered 1-11 months, and
     the month set is coherent (a `lumpy_or_batched` regime says it is not, and that is a
     Return A monthly-file diagnostic so it only vetoes a Return A row).
   * `usable_as_observed` — every other supported row.

   The two are mutually exclusive by construction. Coverage is read off
   `preferred_months_reported` rather than off lane family, which is what lets the NIBRS
   lane be partial and lets a zero offense inside a partial year stay a zero over those
   months instead of falling to the fill ladder.

**What this replaced.** Selection used to run per (`ori9`, offense), so one agency-year
could be assembled out of two lanes: 3,213 rows over 459 agencies in 2024 took NIBRS for
some offenses (published as full-year observed zeros) and Return A for others (annualised
×12/months), and 2,595 agencies carried more than one preferred source. Every per-offense
override that patched that is gone with it — `build_prefer_nibrs_mask`'s
`srs_regime_inferior` arm (including its `true_partial` entry), its
"Return A reports exactly 0 while NIBRS has months > 0" arm, and the regime's own
`preferred_source_by_regime` preference term. With coherent zero and partial semantics in
both lanes those arms have nothing left to fix: a zero in the chosen lane is a zero, and a
partial year in the chosen lane annualises. `build_prefer_nibrs_mask` is **gone entirely**:
its last two call sites were the jurisdiction-level constructions in `controls.py` and
`municipal_estimator.py`, and Stage 3's consumption restructure deleted both — the
jurisdiction layer no longer selects a source at all (see Stage 3 below).

2024 preferred population (scope-filtered): 135,996 rows / 19,428 ORIs. Lane split —
CIUS 59,710, SRS 48,138, NIBRS 24,862, state publication 3,245, local publication 41.
Support split — 98,599 usable as observed, 16,713 true partial. Lane-choice reasons per
agency-year: publication-lane standing 8,661, only lane present 3,843, standing lane
order 3,582, greater month coverage 3,320, published-NIBRS corroboration 18, manual
override 3, greater observation weight 1.

## 1.6 Invariants currently enforced (fail-closed)

| Assertion | Where | What it forbids |
|---|---|---|
| `_assert_no_negative_counts` | `observations.py` | a negative count in the written panel (Return A adjustment residues are clamped and recorded first; a negative on any other lane fails at the clamp) |
| `_assert_fbi_lane_offense_sets_are_complete` | `observations.py` | a Return A or NIBRS agency-year missing an offense row, which would make an absent offense indistinguishable from a zero |
| `_assert_twin_resolution_is_a_function` | `agency_identity.py` | folding two live NIBRS ORIs into one identity, or one variant into two canonical ORIs |
| `_assert_preferred_metadata_matches_the_selected_lane` | `source_selection.py` | a preferred count and its months coming from different lanes (the Kenedy failure) |
| `_assert_preferred_lane_unity` | `source_selection.py` | an agency-year assembled out of more than one lane, except an annotated supplement |
| `_drop_silent_agency_estimates` | `trend_fills.py` | an estimate row for an agency with no current report and no evidence the fill ladder will stand behind — no peer median, no fabricated zero, and no reference older than `FILL_MAX_REFERENCE_AGE_YEARS` |
| `_drop_superseded_ori_estimates` | `trend_fills.py` | any estimate row for an ORI whose municipal footprint a live successor already covers |
| `_assert_no_fill_where_the_chosen_lane_reported` | `trend_fills.py` | a fill for any offense whose chosen lane covered all 12 months (excludes masked-gap-reclassified rows and partial-uplift sanity caps) |
| `_assert_estimates_finite_and_nonnegative` | `trend_fills.py` | NaN / negative `estimated_count`, `agency_adjustment_count`, `reported_count_current` |
| `_check_fill_mass_against_baseline` | `trend_fills.py` | aggregate fill-mass regression beyond the tolerance in `configs/agency_fill_mass_baseline.json` |
| `_assert_agency_mass_flows_through_one_lane` | `benchmark_imputation.py` | an ORI sized by the agency estimator (own report or own-history fill) AND treated as silent territory for benchmark imputation |
| regime/source override validation | `reporting_regimes.py` | unknown regime or source values, and duplicate override keys |
| stage write lock | `stage_locks.stage_write_lock` | concurrent/blocked writes of the observations and regimes artifacts |

Structural invariants that hold but are **not asserted**: unique primary key;
`months_reported ∈ [0,12]`; `months_reported + months_missing = 12`.

### The three lanes an agency's territory can be sized through

Mutually exclusive by construction and asserted to be so:

1. **Its own report** — a usable or true-partial target-year observation.
2. **Its own recent history** — a fill, but only if the last usable report is within
   `FILL_MAX_REFERENCE_AGE_YEARS` (2) of the target year. The ladder's own carry-forward
   and trend rungs already operate over that span; beyond it the pipeline would not be
   extrapolating a reporting gap but reanimating an agency that stopped reporting.
3. **The state benchmark** — `benchmark_imputation`, for territory whose silent agency is
   still on the FBI roster. An agency that is off the roster AND has neither a current
   report nor a reference inside the recency bound is dead: no fill, no imputation, and
   its territory falls to the covariate model with the state total conserved by raking.

## 1.7 Where the current data does not satisfy §1.2–§1.6

Measured over the full 2018-2024 population; row-level enumerations of the 2026-07-29
audit are in `state/qa/stage1_screen/` and counts in `stage1_screen_summary.json`. The
rule-shaped classes that audit found are closed — re-measured against the rebuilt panel,
preferred panel and 2024 agency estimates:

| class | before | after |
|---|---|---|
| same-stem identical-vector twin groups (2024) | 125 groups / 18,579 counts carried twice | **0** |
| NIBRS agency-years submitted but absent from the lane | 8,306 (1,631 in 2024) | **0** |
| NIBRS partial years read as complete (2024) | 8,006 rows / 1,410 agencies | **0** |
| zero offenses inside a partial year sent to the fill ladder (2024) | 3,439 rows | **0** |
| agency-years assembled from more than one lane (2024) | 2,595 agencies | **55 agencies / 109 rows**, every one an annotated publication-lane supplement |
| own-history fills on agencies past the recency bound | 1,935 agencies / 155,789 counts | **0** |
| superseded ORIs resurrected by fill | 23 ORIs / 32,068 counts | **0** (20 still detected; they carry no estimate row) |
| negative counts | 29 | **0** (clamped and recorded) |
| FBI-lane agency-years with fewer than 7 offense rows | 0 | **0**, now asserted |
| panel `county_fips` sentinels / retired GEOIDs | 98 / 525 rows | **0 / 0** |
| out-of-scope ORIs reaching the 2024 agency estimates | 42 ORIs / 62,158 counts | **0** |

What remains, by construction rather than by defect:

* **794 publication-lane agency-years carry fewer than seven offense rows** (CIUS 700,
  state publication 92, local 2). These are withheld or suppressed figures, which are
  missing rather than zero — see §1.2 — and §1.5's supplement rule is how they are read.
* The **ad-hoc classes the audit routed to per-case review are untouched**: ambiguous
  same-stem twins with differing counts (a2, 71 groups) and different-stem identical
  vectors (a3, 70 groups); published full-year zeros (b2, 314 in 2024); token reporters /
  structural zeros (c, ~700); and the off-roster half of the defunct population, which
  this batch routes to the dead predicate rather than adjudicating agency by agency.

---

# Stage 3 — jurisdiction targets and the benchmark identity

Scope: everything between Stage 1's per-agency target-year estimates and the finished
per-jurisdiction × offense control that Stage 4 allocates. Agency→footprint resolution
(the crosswalk) is Stage 2 and is read, not built, here; within-jurisdiction allocation is
Stage 4.

Files: `src/crimerisk/jurisdiction_targets.py` (the skeleton and the aggregation),
`src/crimerisk/controls.py` (the control panel, the build order, the CDE reconciliation),
`src/crimerisk/benchmark_imputation.py` (silent territory).

| Artifact | Builder | Path |
|---|---|---|
| jurisdiction ownership / control skeleton | `jurisdiction_targets.build_jurisdiction_ownership` | in-memory |
| ownership exclusions | `jurisdiction_targets.build_ownership_exclusions` | `state/controls/jurisdiction_ownership_exclusions_<year>.parquet` |
| jurisdiction-year estimates (2018-2024) | `jurisdiction_targets.build_jurisdiction_year_estimates` | `state/controls/jurisdiction_year_estimates.parquet` |
| jurisdiction controls | `controls.build_jurisdiction_controls` | `state/controls/jurisdiction_controls_<year>.parquet` |
| state comparison vs FBI CDE | `controls.build_state_control_comparison` | `state/controls/state_control_comparison.parquet` |
| benchmark imputation units / identity / diagnostics | `benchmark_imputation.build_benchmark_imputation` | `state/controls/benchmark_imputation_*_<year>.*` |

## 3.1 There is no jurisdiction-level estimator

Every target amount published at this stage is a weighted sum of Stage-1 per-agency
estimates:

```
pre-imputation target(jurisdiction, offense) = Σ_agencies  estimated_count × crosswalk weight
```

for all three lanes — `municipal`, `state_nonmunicipal_remainder` and
`statewide_overlap_layer`. No source is preferred here, no usability rule is evaluated
here, and no fill is computed here. Selection and filling are agency facts and neither
commutes with aggregation (`fill(Σ agencies) ≠ Σ fill(agency)`; one reporting agency can
make a jurisdiction look current while another inside it is silent), so doing either after
the sum produces a different — and unaccountable — estimand. Jurisdiction-level models may
allocate these amounts spatially at Stage 4; they may not resize them.

**What this replaced (v21).** `municipal_estimator.py` re-derived a per-offense source
preference, its own `usable_as_observed` rule and its own seven-rung fill ladder from
`jurisdiction_year_observations.parquet`, and `jurisdiction_estimator.py` ran a second,
near-duplicate ladder plus two target-year override patches; the municipal result was
written onto the control unconditionally. All five defect classes Stage 1 closed at the
agency-year were live again there (partial years read as complete, fabricated peer
anchors, fills past the recency bound, per-offense lane forks, silent agencies
re-animated), the overlap layer got no override at all and dropped 21,757 counts of agency
mass, and a municipal fill could lock a control before benchmark eligibility was
evaluated. Both modules are deleted, as is `build_prefer_nibrs_mask` and the
`panel_guardrails` spike suppressor that only ever ran on their panels.

## 3.2 The skeleton decides the control universe, not the estimate table

`build_jurisdiction_ownership` runs before any mass is aggregated. A jurisdiction owns
territory, and therefore gets a control row per offense, when:

* `municipal` / `state_nonmunicipal_remainder` — it holds block groups in
  `block_group_to_jurisdiction_crosswalk`, the same crosswalk that supplies benchmark
  exposure;
* `statewide_overlap_layer` — at least one agency routes to it in the agency crosswalk.
  This lane holds no block groups by design; its footprint is resolved at allocation.

Everything else is **excluded and enumerated** in
`jurisdiction_ownership_exclusions_<year>.parquet` with its reason, its agency links and
the agency mass it carries. Ordering is the point: a jurisdiction whose only agency is
silent has a control row at zero, which is what lets `benchmark_imputation` see it. The
alternative — deriving controls from rows that exist — makes silence indistinguishable
from absence and re-creates the under-imputation defect with no fill anywhere in sight.

## 3.3 Provenance is composed, never chosen

The control keeps, per row: each source lane's own crosswalk-weighted reported rollup
(`reported_count_*`), each source lane's share of the target (`target_count_from_*`), and
the three estimate-class components (`observed_component_count`,
`partial_component_count`, `fill_component_count`). `preferred_source`,
`dominant_reporting_regime`, `quality_tier_preferred`, `relationship_type_preferred` and
`overlap_subtype_preferred` survive as **descriptive labels only** — mass-weighted
dominant labels for the published surface. Nothing in this stage or downstream of it may
make a selection or usability decision from them: a dominant label conceals mixed agency
provenance by construction, which is how 289 rows carrying 201,621 counts came to publish
one lane's quality tier against another lane's count.

`quality_tier_preferred` is now the Stage-1 tier ladder applied to the jurisdiction's own
mass-weighted month coverage, so the tier and the count describe the same thing.

## 3.4 The control row

```
estimated_count_ags_core = Σ(agency estimate × weight) + benchmark_imputed_count
adjusted_count_ags_core  = estimated_count_ags_core
adjustment_total        = target − reported
identity_resolution_adjustment_count =
    −Σ(preferred reported count × weight) for superseded ORIs in the succession ledger
partial_reporting_uplift_count = Σ(agency adjustment × weight) where the agency source is
                                 true_partial_month_ratio
current_year_fill_count        = Σ(agency adjustment × weight) where the agency source is
                                 not lane-grounded
adjusted_count_ags_core =
    reported_count_preferred + identity_resolution_adjustment_count
    + partial_reporting_uplift_count + current_year_fill_count + benchmark_imputed_count
needs_partial_reporting_uplift = uplift > 0
needs_zero_month_fill          = reported ≤ 0 and target > 0
estimate_source ∈ {agency_rollup_observed, agency_rollup_partial_uplift,
                   agency_rollup_fill, no_agency_evidence}   (target year)
                 {agency_reported_rollup}                     (history years)
estimate_confidence = high | medium (uplift present) | low (fill, or no evidence)
estimated_from_panel = the row carries uplift/fill mass, OR carries no agency evidence
```

The uplift/fill split is read off the agency estimate class rather than off a
jurisdiction-level months column. The old flag keyed on
`mean_months_reported_preferred`, an unweighted mean over contributing agencies, so it
over-fired on complete cities sharing a jurisdiction with a silent agency and under-fired
on the remainder pools, where 3,231 agencies share 47 rows and 75,865 counts of genuine
partial-year uplift published as `current_year_fill`.

History years (2018 … target−1) carry the reported rollup and nothing else: a year in
which an agency did not report is a year with no jurisdiction-level evidence, and
inventing one would put the deleted ladder back into the reference set.

## 3.5 Build order

`controls.build_controls_bundle`, in this order and only this order:

1. Stage-1 consumption — agency panel, ORI succession ledger, agency target estimates
   (one `Stage1Consumption` object, so the fill lane and the benchmark lane cannot see
   different ledgers).
2. ownership skeleton + exclusion artifact.
3. jurisdiction-year estimates (target year consumed, history rolled up).
4. pre-imputation controls (the target-year slice of 3).
5. benchmark imputation, evaluated against the skeleton.
6. imputed mass landed on the controls.
7. state comparison against the FBI CDE series.

## 3.6 Benchmark imputation: eligibility before mass

The accounting identity per state × offense is unchanged (see the module docstring):
`residual = max(0, CDE − locked)`, `imputed = modeled pool × min(1, residual / pool)`.
What changed is how a unit becomes eligible. Eligibility is now stated entirely over the
**agency ledger**:

```
silent unit = no supported agency AND no fill-covered agency (over links with weight > 0)
              AND at least one eligible-silent agency AND exposure > 0
```

and "its control row is empty" is asserted as a **post-condition**
(`_assert_every_silent_unit_lands_on_exactly_one_empty_control_row`) rather than used as
the rule. It used to be the rule (`locked_total ≤ 1e-9`), which is what made the Jackson
MS shape possible: the municipal ladder filled the city from a 2019 reference the agency
estimator had already refused as stale, the fill locked the control, and the unit was
silently dropped here.

Two consequences recorded rather than hidden:

* `build_silent_agency_ledger` now receives the **succession ledger**, so `is_superseded`
  is live in the production path instead of always `False`.
* units holding both a silent primary agency and an agency the fill ladder already sizes
  are **not** imputed — the exposure cannot be split between them — and are counted in the
  diagnostics as `partially_covered_units_not_imputed`.

## 3.7 Invariants enforced at this stage (fail-closed)

| Assertion | Where | What it forbids |
|---|---|---|
| `_assert_crosswalk_weights_partition_every_agency` | `jurisdiction_targets` | crosswalk weights that do not sum to 1 per ORI (agency mass lost or duplicated) |
| `_assert_every_agency_estimate_lands_on_the_skeleton` | `jurisdiction_targets` | positive agency mass routed to a jurisdiction with no control row |
| `_assert_row_identity` | `jurisdiction_targets` | `target ≠ reported + uplift + fill` on any row |
| `assert_agency_mass_equals_control_mass` | `jurisdiction_targets` | pre-imputation control mass ≠ agency-estimate mass, per state × offense × lane |
| `_assert_every_silent_unit_lands_on_exactly_one_empty_control_row` | `benchmark_imputation` | an eligible silent unit with no landing row, more than one landing row, or non-zero pre-imputation locked mass |
| `_assert_lane_partition` | `benchmark_imputation` | block-group `allocation_share` not summing to 1 across the two geometry lanes |
| `_assert_agency_mass_flows_through_one_lane` | `benchmark_imputation` | an ORI sized by the agency estimator AND treated as silent territory |
| `assert_benchmark_imputation_invariants` | `benchmark_imputation` | imputed mass in a conflict cell, above the residual, non-finite, duplicated, or claimed by both lanes |
| mass-landing check | `benchmark_imputation` | imputed mass that does not land on a control row |
| `check_benchmark_mass_against_baseline` | `benchmark_imputation` | national imputed mass rising beyond the ratchet tolerance |
| `stage_write_lock("controls")` | `stage_locks` | concurrent or blocked writes (the `municipal_estimates` stage no longer exists) |

---

# Stage 4 — within-jurisdiction allocation: the share basis

Scope: this section covers only what the Stage 4/5 rule batch established. The full
descriptive stage contract is `state/qa/stage4_screen/CONTRACT.md`.

Files: `src/crimerisk/crosswalk_shares.py` (the block-group share basis),
`src/crimerisk/allocation.py` (the lanes, the registries, the displacement).

## 4.1 One share basis per block group, and a recipient floor

`allocation_share` is the fraction of a block group's exposure a jurisdiction owns, and
every lane multiplies it by the block group's activity weight (`bg_weight ×
allocation_share`). Two rules govern it, both in
`crosswalk_shares.normalize_block_group_allocation_shares`:

1. The ladder `pop_share → housing_share → block_share → aland_share → allocation_share`
   is resolved **once per block group**, not per row. The first rung whose block-group
   total is positive is the basis for every row in that block group; a fragment that is
   zero on that basis is not a recipient. Applied per row, the ladder let a populated
   fragment be measured by population while a zero-population fragment in the same block
   group was measured by its block count, and summed the two as commensurable: 13,811
   block groups (5.8%) normalised a mixed basis and 14,584 crosswalk rows with zero resident
   population held a positive share inside a populated block group. Both are 0 after the fix.
2. A fragment covering less than `CROSSWALK_MINIMUM_RECIPIENT_SHARE = 0.02` of its block
   group on that basis is **not an independent recipient**. Its share is routed to the
   block group's remaining recipients, and its own jurisdiction's target spreads over the
   block groups it materially covers — mass is conserved because the component share is
   normalised inside the jurisdiction, not inside the block group. Basis for the value:
   below 2% the fragment is a block-assignment boundary artifact (median 3 census blocks
   carrying 10 residents, against a median block group of 22 blocks, so under half of one
   block's share of the block group), and nothing in its geometry supports the uniformity
   that `bg_weight × share` assumes, while the mass it delivers is set by its
   jurisdiction's total rather than by the fragment.

The floor never strands mass: it is not applied to the last positive recipient of a block
group, of a jurisdiction, or of a county's non-municipal remainder.

Three audit columns are written beside the share: `allocation_basis` (the block group's
rung), `allocation_share_before_recipient_floor`, and `allocation_recipient_status` ∈
{`recipient`, `zero_on_block_group_basis`, `below_minimum_recipient_share`,
`floor_exempt_only_recipient_in_block_group`, `floor_exempt_only_support_for_jurisdiction`,
`floor_exempt_only_support_for_county_remainder`}. Neither the basis nor the status is
carried past `allocation._load_bg_crosswalk`: the consolidated-footprint lane unions rows
from two jurisdictions, and carrying either column through would force a dominant label
onto a mixture.

`assert_allocation_shares_conserve` fails the build unless the normalised shares sum to 1
in every block group. Before the floor this held structurally and was unasserted.

## 4.2 What a custom footprint's `weight_share` measures

`configs/overlap_custom_footprints.csv` carries `weight_share_basis` per row, declared and
never inferred from the geometry-source string, with one basis enforced per (ori, state):

| value | what the share is | how the lane uses it |
|---|---|---|
| `resident_population` | a 2020 resident-population or housing-unit share of the footprint (tribal AIANNH block-assignment footprints, state-police post footprints) | `bg_weight × share`, normalised inside the pool — the same activity basis every county lane uses |
| `activity_or_area` | already an activity or exposure measure (station boardings, annual passengers, LandScan daytime population, per-station equal share), or a deliberate area apportionment onto parcels with no residents (airfields, port property) | verbatim; multiplying by `bg_weight` would apply an activity term twice |

Fail-safe, not fail-open: a `resident_population` footprint whose whole support carries zero
activity weight for an offense keeps the declared population apportionment rather than
stranding the agency's mass.

## 4.3 The concurrent-jurisdiction carve-out

Exclusive (remainder-displacing) footprints are the default. `configs/`
`concurrent_jurisdiction_carveouts.csv` names the counties where displacement instead
reverts to **shared overlap** — the footprint keeps its mass and its placement but stops
displacing the non-municipal remainder — because displacement removes more than half of a
*reporting* remainder agency's block-group exposure there. Every row carries
`reviewer_note = concurrent_jurisdiction_unresolved` and the loader accepts no other value:
a carve-out is an unresolved PL-280 adjudication, not a decided treatment. A carve-out
county that no displacing footprint touches fails the build rather than sitting inert.

---

# Stage 5 — counts → published rates and indices: eligibility and vocabulary

Scope: this section covers only what the Stage 4/5 rule batch established. The full
descriptive stage contract is `state/qa/stage5_screen/CONTRACT.md`.

## 5.1 The two suppression vocabularies

`estimate_mode_{offense}` is the **display** vocabulary. It carries five values
(`count_derived`, `non_residential`, `special_use`, `insufficient_exposure`,
`vehicle_denominator_invalid`), they are what the tiles encode as integer codes, and the
tile builder fails closed on any value it does not know.

`denominator_reason_{offense}` and `resident_denominator_reason_{offense}` are the
**mechanism** vocabulary. They are finer than `estimate_mode` by one value:
`insufficient_ambient_exposure` names the ambient-blind footprint class (§5.2), which
displays under `insufficient_exposure` because "too little exposure for a per-person rate"
is exactly true of it. Both arms now also carry `non_residential`; before this batch
`denominator_reason` had no such value at all, so 2,306 block-group rows per offense (781
tract rows, 771 on the resident arm) read `publishable` next to a null index. Both values
are asserted in `scripts/diagnostics/validate_release_outputs.py`.

## 5.2 Ambient-blind custom footprints

A casino, trust-parcel or campus footprint can be placed correctly and carry real counts
while its published denominator is its resident population and the people who generate the
counts are visitors who appear in neither LODES nor LandScan. Three measured conditions,
per block group and offense, all of them quantities rather than labels:

```
footprint_derived_count_share_{o} > FOOTPRINT_DERIVED_MASS_SHARE_FLOOR (0.5)
exposure_proxy_2024 <= population                       (no ambient lift at all)
100000 × poisson_lower(count_o) / population
    >= AMBIENT_BLIND_FOOTPRINT_RESIDENT_RATE_RATIO (3.0) × national resident rate
```

The third condition is measured against the national resident rate over the rows publishable
under the pre-existing floors, so the rule is one pass and not circular in its own
suppression. It is measured on the **lower bound of the count's exact-Poisson interval**, not
on the point count: on the point count the rule fired on 66 murder and 38 rape cells whose
median flagged count was 0.27 and 2.30, and 17 tracts would have lost a published murder index
because a metro-station footprint supplied more than half of a fractional murder allocation.
"The residents cannot account for the counts" is only evidence when the counts can be told
apart from a much smaller number, and the interval already published on every row is the
surface's own device for saying so — no new threshold is introduced.

Measured on the rebuilt surface the rule reaches **23 block groups / 34,643 residents / 2,221
expected counts**, in 4 offense cells for rape, 6 robbery, 8 aggravated assault, 1 burglary, 4
larceny, 2 MVT, and **0 murder**. Where it fires the primary and resident indices are suppressed,
the expected count and the crime density still publish, `estimate_mode` reads
`insufficient_exposure` and `denominator_reason` reads `insufficient_ambient_exposure`. A cell that is both below the
plain exposure floor and ambient-blind reports the floor: the floor is the harder structural
fact and does not need the footprint. Published beside the flag:
`footprint_derived_count_{o}` (the compositional mass, summed at tract rollup) and
`footprint_derived_count_share_{o}` (derived where used, at both levels).

## 5.3 The transient-exposure guard

```
transient_exposure_likely_{o} = population > 0
                              & exposure_proxy_2024 / population >= 5
                              & index_{o}_resident >= 1000
```

Advisory: it gates no publication and moves no point value. Two corrections against the
shipped form. The `households_total >= 10` term is gone — every one of the 290 candidate
cells it excluded is non-residential, and a non-residential cell publishes no index, so the
index condition was already False there and the term blocked nothing. And the index
condition now reads the **resident** index: the ratio's own complaint is that the resident
denominator understates the population at risk, but murder, rape, robbery, aggravated
assault and larceny already publish against person exposure, so the primary index had
already absorbed the transience the ratio indicts. The ratio itself is published as
`transient_exposure_daytime_to_resident_ratio` (one column: the ratio does not vary by offense).

Measured on the rebuilt surface: the shipped definition reaches **82 block groups**, the
corrected one **2,147 block groups / 1.81M residents** (robbery 1,040 cells, aggravated assault
1,057, burglary 725, larceny 1,450, MVT 457). The households term blocks **0** of them, which is
what made it dead code.

`apply_rare_offense_tract_support` clears the flag for murder and rape at block-group
support, because it nulls the resident index the flag reads. Before, 88 of the guard's 162
firings (murder 86, rape 2) sat on an index no consumer could see; the rebuilt surface carries
0 rare-offense firings at block group.

The flag reaches one consumer: `confidence.py` appends a `transient_exposure_likely` token to
`confidence_reasons_{o}`. It does not enter `confidence_tier`, and it gates no publication.

## 5.4 The resident-aggregate boundary

`SURPRISE:` the "all-or-null aggregate" rule stated in `docs/STATE.md`'s *Published field
and index policy* does not describe the block-group surface, and this is a boundary rather
than a defect.

`index_total_part1_resident`, `index_personal_part1_resident` and
`index_property_part1_resident` are **count-derived** aggregates: each is the summed count
over its offense set divided by the resident denominator, indexed to the national rate of
the same construction. The per-offense resident index only ever served as a publication
mask for them. They are computed inside `_finalize_output`, and
`apply_rare_offense_tract_support` runs afterwards and nulls the murder and rape *resident*
component indices at block-group support without re-deriving the aggregates that used them
as a mask. Result: `index_total_part1_resident` publishes on 234,888 block-group rows whose
murder/rape resident component is null, and `index_personal_part1_resident` on 235,160.

The published values are correct — they are the count sum over the resident denominator, and
the murder/rape counts are retained on all 238,193 rows — but a consumer who tries to verify
all-or-null from published fields finds 235k counter-examples. The three **primary**
composites (`index_total_primary_event_weighted`, `index_total_equal_offense`,
`index_total_harm`) are re-derived inside `apply_rare_offense_tract_support` from tract-support
murder/rape terms and are internally coherent: 0 rows published with a null component, 0 rows
nulled without one.

So the boundary is: **the primary composites are all-or-null over their seven components; the
resident Part 1 aggregates are count-derived and are all-or-null over the seven resident
*denominator* eligibilities, not over the seven resident component indices.**
