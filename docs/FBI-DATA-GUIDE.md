# FBI Crime Data Guide

Canonical reference for what the FBI's crime-data terms actually mean, how the public
releases are actually produced, and what this repo has empirically verified against the
frozen data under `data/`. This document is authoritative within this repo: where external
prose (FBI summaries, AGS methodology text, blog posts, secondary analyses) conflicts with
this document, this document wins unless new evidence is produced and recorded here.

> **Scope note for the submission package.** This guide describes the full development
> workspace. File paths it cites under `data/`, `state/`, and `scripts/diagnostics/`
> (including the conversion verification and tuning scripts) refer to that workspace and
> are not all included in the clean submission package. The NIBRS/SRS conversion checks
> were run during development; the package ships the summarized methodology here plus the
> final release outputs, not the full tuning workspace or its intermediate artifacts.

Every load-bearing claim is tagged with its evidence class:

* `[verified]` — measured directly against the frozen data in this repo (June 2026,
  reproducible via `scripts/diagnostics/verify_fbi_conversion_rules.py`)
* `[fbi-doc]` — stated in an official FBI publication (cited)
* `[kaplan]` — stated in Jacob Kaplan's UCR book (`docs/ucrbook_full.md`, line refs)
* `[inference]` — our best reading, not directly measured

---

## 1. Glossary: similar-sounding terms that are not the same thing

| Term | What it is | What it is not |
|---|---|---|
| **UCR (Uniform Crime Reporting Program)** | The FBI umbrella program. Contains SRS, NIBRS, hate crime, LEOKA, use-of-force, etc. | Not a dataset. Not a format. Not "the thing NIBRS replaced" — NIBRS is *inside* UCR. |
| **SRS (Summary Reporting System)** | The legacy *submission format*: aggregate monthly offense counts per agency. | Not a publication. Not synonymous with "the Return A file" (see below — that file now contains mostly converted NIBRS). |
| **Return A** | The specific SRS monthly form for "Offenses Known and Clearances by Arrest" — the Part I offense counts. "Return A", "Offenses Known", and "the SRS crime data" refer to the same collection. | Not a frozen product. The master file is revised as agencies resubmit. |
| **NIBRS (National Incident-Based Reporting System)** | The incident-level *submission format*: one record set per incident with offense/victim/offender/property segments. | Not a separate program from UCR. Not the only thing the FBI accepts (native SRS was still accepted through at least 2024). |
| **NIBRS segments** | The relational pieces of a NIBRS submission: batch header (agency-year enrollment), administrative (incident), offense, victim, offender, arrestee, property. | The offense segment alone is not "the NIBRS data" — counting anything SRS-comparable requires the victim and property segments too (§4). |
| **CIUS ("Crime in the United States")** | The FBI's annual *publication/table suite* (Tables 1–7 estimates; Table 8/9/11 "Offenses Known" agency rows; etc.). Recent years also branded **RCN ("Reported Crimes in the Nation")**. | Not a database. Not "SRS only." A frozen snapshot, not a living file (§5). |
| **CDE (Crime Data Explorer)** | The FBI's portal/API/database that exposes UCR collections and derived tables, including bulk CSV downloads. | Not a single table. Preserves the SRS-vs-NIBRS distinction internally even where the front end blurs it. |
| **`estimated_crimes` (CDE estimates file)** | CDE's national/state annual *estimated* totals: reported data plus FBI estimation for non-reporting and partially-reporting agencies. | Not a sum of agency rows. Never use it to define a local remainder by subtraction (SPEC doctrine). |
| **Kaplan files** | Jacob Kaplan's openICPSR concatenations of the FBI's raw master files (Return A 1960–2024, NIBRS segments 1991–2024). These are the FBI's own master-file contents, restacked — not a third-party reinterpretation. Exact chain and vintage: §2.1. | Not a snapshot of what was published in any CIUS year — they carry the *revised* master-file values as of the build vintage. |
| **Hierarchy rule** | SRS rule: in a multiple-offense incident, classify the incident by the most serious Part I offense. | Not the main SRS/NIBRS counting difference (§4.3 — it is a ~2.4% effect). Coexists with per-victim counting; it ranks offenses, it does not set the counting unit. |
| **Hotel rule** | SRS burglary rule for multi-unit lodging burglarized under single management. In NIBRS→SRS conversion this is driven by `number_of_premises_entered` gated on lodging-type locations. | Not applied to all multi-premises incidents. |
| **Part I / "legacy seven"** | Murder, rape, robbery, aggravated assault, burglary, larceny-theft, motor vehicle theft (arson is Part I but excluded from CrimeRisk-style products for reporting inconsistency). | — |
| **Revised rape definition (2013)** | UCR rape since 2013 = NIBRS 11A (rape) + 11B (sodomy) + 11C (sexual assault with an object). | Not NIBRS code 11A alone. Mapping only `sex offenses - rape` undercounts UCR-definition rape by ~25% `[verified]`. |

---

## 2. The actual data flow

```text
Agency submissions, 2024 [fbi-doc: RCN 2024 FAQ]
  14,601 agencies via NIBRS   (87.2% of enrolled-agency population)
   2,074 agencies via SRS     (+8.4%; total coverage 95.6%)
        |
        v
FBI converts every NIBRS submission into Return A (SRS-format) rows,
applying the full SRS counting rules (verified rule-by-rule in §4)
        |
        v
Return A MASTER FILE  <-- a LIVING dataset: revised as agencies resubmit
  = what Kaplan concatenates; = what this repo loads as srs_return_a_annual
        |
        +--> CIUS / RCN published tables  <-- FROZEN snapshot at publication time
        |      = what this repo loads as cius_publication_annual
        |
        +--> CDE estimated totals (estimated_crimes CSV)
               = master-file data + FBI estimation for missing agencies/months
```

The single most important fact, because almost all outside prose gets it wrong or buries
it: **the conversion happens upstream, inside the Return A master file itself.** There is
no public "dataset of agencies that submitted SRS." Anyone consuming the released Return A
data — the FBI's own CIUS tables, Kaplan, AGS, this repo — is consuming FBI-converted
NIBRS for the large majority of agencies, whether or not they do any "NIBRS handling"
themselves. `[kaplan: ucrbook_full.md ~154–159: "when agencies report only NIBRS the FBI
converts and releases that data as its SRS version"; verified in §3]`

### 2.1 Exact provenance of the files this repo measures against

When this guide says "the Return A file", "the NIBRS segments", "CIUS", or "the CDE
estimates", it means these specific frozen files and nothing else:

| Repo path | What it is, exactly | Producer chain | Vintage |
|---|---|---|---|
| `data/SRS-Kaplan-1960-2024/offenses_known_parquet_1960_2024_year.zip` (+ `_month`) | The Return A ("Offenses Known and Clearances by Arrest") master-file data, all agencies, concatenated 1960–2024 | FBI Return A master files → Jacob Kaplan's openICPSR concatenated release (restacking + column-name standardization, not value re-derivation) → frozen here. Provenance constant: `kaplan_openicpsr_srs_return_a`. | Kaplan build **2025-08-15** (internal file dates) — days after the FBI's early-August 2025 RCN 2024 publication; staged in repo 2025-12-08 |
| `data/NIBRS-Kaplan-1991-2024/*_segment_parquet_1991_2024.zip` | The NIBRS master files, per segment (batch header, administrative, offense, victim, offender, arrestee, property, Group B), concatenated 1991–2024 | Same Kaplan openICPSR chain | Kaplan build 2025-08-15; staged 2025-12-08 |
| `data/FBI-CIUS-Annual/<year>/raw/offenses-known-to-le-<year>.zip` | The FBI's own published "Offenses Known to Law Enforcement" table bundles (Table 8/9/11 family) for publication years 2018–2024 | FBI CIUS/RCN publication downloads, taken directly from the FBI | staged 2026-03-22 |
| `data/FBI-CDE-Estimates-1979-2024/estimated_crimes_1979_2024.csv` | CDE's national/state estimated annual totals (the "Summary (SRS) data with estimates" bulk download) | FBI Crime Data Explorer, taken directly | staged 2025-12-30 |
| `state/cache/{offenses_known_year,nibrs_offense,nibrs_batch_header,...}/` | Per-year extractions of the zips above, made by this repo's loaders | Derived from the rows above — not independent sources | — |

Three clarifications that matter:

1. **"The Return A file" here is the Kaplan openICPSR file — not a CDE API endpoint and
   not the CIUS tables.** It sits one repackaging step away from the FBI's raw master
   files. That step's faithfulness is not assumed; it is what §5 empirically validates:
   at matched vintage the file's values equal the FBI's published CIUS local rows in
   99.88% of comparable cells, so the file behaves as the FBI master-file content. Treat
   any future re-download (Kaplan rebuild or FBI CDE master-file download) as a *new
   vintage* and expect revisions (§5).
2. **The master-file vintage is the Kaplan build date (2025-08-15), not the repo staging
   date.** Every "living file vs frozen publication" statement in this guide is relative
   to that vintage. It postdates the RCN 2024 publication by days, which is exactly why
   the 2023/2024 CIUS agreement is ~100% while 2021/2022 agreement has drifted.
3. **Panel filtering**: the §3 presence table and all dual-agency tests run on this
   repo's local-agency panel (agency types local police department and constable/marshal),
   not the full file. The national reported-vs-estimated sums in §6 use the full,
   unfiltered Return A file. Counts here are therefore not comparable to FBI
   all-agency-type counts (sheriffs, state agencies, etc. are excluded from the panel).

---

## 3. Verified: the Return A file is mostly converted NIBRS

"Return A file" = the Kaplan openICPSR concatenation of the FBI Return A master files,
vintage 2025-08-15 (exact chain in §2.1). All measurements on this repo's local-agency
panel (`state/observations/agency_year_observations.parquet`, built from that file),
agencies with `months_reported > 0`.

| Year | Return A agencies | also in NIBRS rollup | SRS-only | NIBRS-only |
|---|---|---|---|---|
| 2018 | 16,278 | 6,927 (43%) | 9,351 | 126 |
| 2019 | 16,177 | 8,255 (51%) | 7,922 | 282 |
| 2020 | 15,224 | 9,768 (64%) | 5,456 | 115 |
| 2021 | 12,459 | 11,734 (94%) | 725 | 233 |
| 2022 | 14,639 | 12,323 (84%) | 2,316 | 277 |
| 2023 | 14,612 | 12,412 (85%) | 2,200 | 277 |
| 2024 | 14,805 | 12,737 (86%) | 2,068 | 275 |

* The 2021 row is a natural experiment: the FBI accepted **zero** native SRS submissions
  for 2021 `[fbi-doc]` `[kaplan]`, yet 12,459 local agencies have populated Return A rows —
  conversion is the only way those rows can exist. `[verified]`
* Of the 725 "SRS-only" 2021 agencies, 74.5% appear in the NIBRS **batch header** for 2021:
  they are NIBRS reporters whose incidents don't surface in an offense-segment rollup
  (zero-incident agencies produce batch-header rows but no offense rows; 26.6% are in the
  batch header with zero Part I). `[verified]` Presence in the offense segment is therefore
  a **lower bound** on NIBRS participation; use the batch header for participation.
* The 2024 SRS-only count (2,068, local-only panel) is consistent in magnitude with the
  FBI's "2,074 agencies submitted via SRS" `[fbi-doc: RCN 2024 FAQ]`. Do not treat these as
  the same set — different universes — but the scale matches.
* The FBI resumed accepting native SRS in 2022 after the 2021 NIBRS-only attempt.
  `[kaplan: ~148–159]` 2024 is explicitly a mixed NIBRS + native-SRS year. `[fbi-doc]`

---

## 4. Verified: the FBI's NIBRS→SRS conversion rules, rule by rule

Measured on 2024 dual-reporting agencies (those with both a populated Return A row and a
NIBRS rollup), comparing the FBI's converted Return A totals against recomputations from
the raw NIBRS segments. Ratio = Return A total ÷ recomputation; 1.0 means the rule
reproduces the FBI's conversion.

| Offense | Counting unit (verified) | Ratio | Naive incident-count error |
|---|---|---|---|
| Murder | one per **victim** | 1.006 | −8% |
| Rape | **11A+11B+11C**, one per victim | 1.004 | −25% (11A-only incidents) |
| Aggravated assault | one per **victim** after higher-person/robbery hierarchy suppression | 1.007 | −19% |
| Robbery | one per **incident** (distinct operation), after higher-person hierarchy suppression | 1.015 | ~0 |
| Burglary | one per incident + **hotel rule** premises multiplier, after higher-category hierarchy suppression | 1.016 (bracket 1.048→0.989) | −5% |
| Larceny-theft | one per **incident**, suppressed below motor vehicle theft | 1.008 | ~0 |
| Motor vehicle theft | one per **stolen vehicle** (`number_of_stolen_motor_vehicles`, property segment), above larceny but below burglary/person offenses | 1.012 | −2% |

These are not reverse-engineered from scratch: they are the FBI's published scoring
doctrine — UCR Handbook: crimes against persons scored one offense per victim; crimes
against property one per distinct operation; motor vehicle theft one per stolen vehicle
`[fbi-doc]`. What was *not* documented anywhere usable is whether the FBI's NIBRS-to-SRS
conversion actually implements that doctrine in the released master files (as opposed to
something simpler, like incident counting). That is what the table confirms, rule by
rule. `[verified]`

Notes:

1. **Burglary hotel rule**: counting incidents only gives 1.048; multiplying *all*
   incidents by `number_of_premises_entered` gives 0.989. The production rollup applies
   the multiplier only to qualifying lodging-type locations, caps legitimate observed
   premises at `max(p95, 30)`, and treats `99` as a cap sentinel. `[verified]`
2. **MVT hierarchy position**: the raw per-vehicle comparison gives 0.990, while the
   production Part I rollup lands at 1.012 after hierarchy suppression. Motor vehicle
   theft is above larceny for suppression, but below burglary and person offenses.
   `[verified]`
3. **Hierarchy rule materiality** (2024, all NIBRS agencies): 6,143,512 Part I offense
   rows vs 6,000,230 Part I incidents → counting every Part I offense instead of the
   hierarchy-top offense adds only **+2.4%**; only 2.0% of Part I incidents contain more
   than one Part I category. `[verified]` The popular framing "SRS only counted the most
   serious crime, NIBRS counts everything" describes a ~2% effect at the Part I category
   level. The counting-unit rules above are the 8–25% effects. NIBRS's real value is
   incident-level detail (location, time, victim attributes), not Part I count inflation.

**Consequence for this repo**: the repo's own NIBRS rollup
(`nibrs_srs_equivalent_annual`, `crime/nibrs.py`) implements the rules in this table,
with rule variants checked against the converted Return A on dual agencies
(`scripts/diagnostics/verify_fbi_conversion_rules.py`; report in
`state/qa/fbi_conversion_rules_verification.md`). In 2024, 59,969 dual agency-offense
pairs have exact equality in 98.8% of cells and aggregate Return A / rollup = 1.0092.
Per-offense ratios are murder 1.006, rape 1.004, robbery 1.015, aggravated assault 1.007,
burglary 1.016, larceny 1.008, and MVT 1.012. The previous incident-distinct rollup was
systematically low by the
"naive incident-count error" column above; it was the preferred source for ~1.3% of
observed count volume (NIBRS-only agencies), which is the slice the rebuild moved.
Earlier years fit looser than 2023–2024 because mid-year NIBRS transitioners have
full-year Return A rows but only partial-year NIBRS segments — composition, not rule
error. The SRS-above-NIBRS source priority remains correct: the FBI's converted rows
are the agency's full-year record; the rollup only fills agencies the Return A file
lacks.

---

## 5. Verified: CIUS is a frozen snapshot of a living master file

For agency-offense cells where Return A and our NIBRS rollup disagree, which value does
the published CIUS row equal?

| CIUS year | disagreeing cells | = Return A | = NIBRS rollup | = neither |
|---|---|---|---|---|
| 2021 | 7 | 28.6% | 0% | 71.4% |
| 2022 | 968 | 61.7% | 1.3% | 37.0% |
| 2023 | 424 | **100%** | 0% | 0% |
| 2024 | 390 | **100%** | 0% | 0% |

Across *all* comparable cells (not just disagreeing ones), CIUS-vs-master agreement by
publication year measures vintage drift directly: 2021 73.3%, 2022 85.7%, 2023 99.88%,
2024 99.88% (59,710 cells). `[verified]` The older the publication, the more the living
master file has been revised away from it.

Reading:

* **Current-vintage years**: the CIUS "Offenses Known" agency tables are not a third
  number — they are the Return A file, frozen at publication. (Our master-file vintage —
  Kaplan build 2025-08-15, §2.1 — postdates the RCN 2024 publication by days, hence
  exact agreement for 2023/2024.)
* **Older years (2020–2022)**: the master file has since been revised by resubmissions,
  while the publication stayed frozen — hence cells that match *neither* current source.
  The FBI documents that master data continue to be updated as reports roll in while
  published tables are point-in-time. This also means **small residual divergences between
  any faithful re-derivation (e.g., a counting-rule-correct NIBRS rollup landing within
  ~1%) and a frozen publication are expected and are not evidence of a wrong rule** —
  vintage drift alone produces them.
* **Repo stance**: preferring `cius_publication_annual` over `srs_return_a_annual` means
  preferring the frozen published snapshot over later-revised master data. For 2023–2024
  this is a distinction without a difference; for 2020–2022 it is a real choice
  (~1/3 of disagreeing cells were later revised). We keep the publication-first priority
  because the design target is replicating a publication-anchored product (AGS CrimeRisk),
  and because published rows are the auditable public record.

---

## 6. Verified: where estimation and imputation happen, and how big they are

* **FBI estimation share** (national, Return A reported sum ÷ CDE estimated total):

  | Year | Violent | Property |
  |---|---|---|
  | 2021 | 0.719 | 0.739 |
  | 2022 | 0.964 | 0.950 |
  | 2023 | 0.938 | 0.932 |
  | 2024 | 0.954 | 0.943 |

  In normal recent years the FBI's published estimates sit ~4–6% above the reported master
  file (whole-agency + partial-year estimation); in the 2021 cliff year, ~26–28% of the
  estimate is estimation. `[verified]`
* **Partial-year reporting** is the larger raw surface: ~39–40% of reporting local
  agencies submitted fewer than 12 months in 2020–2024 (34% in 2018, 31% in 2019).
  `[verified]` This is why `months_reported` must be derived from the monthly file's
  `number_of_months_missing` (true-months convention, see SPEC) and why partial years are
  modeled, never mechanically annualized.
* **CDE estimates CSV quirks**: national rows for 2022+ carry comma-formatted numbers
  (`"1,295,605"`) and parse as strings; the 2024 national violent estimate (1,221,345)
  matches the RCN 2024 FAQ exactly. `[verified]`

---

## 7. Claims to reject on sight

Each of these appears in otherwise-credible writing. All are wrong or misleading:

1. **"UCR was replaced by NIBRS."** NIBRS is a collection inside the UCR program. The UCR
   program, the Return A collection, and the CIUS publication all continue.
2. **"CIUS is SRS data."** Current CIUS is built from the Return A master file, which is
   predominantly FBI-converted NIBRS (86% of 2024 local agencies) plus native SRS.
3. **"The FBI stopped collecting SRS in 2021, so there's no 2021 SRS data."** Native SRS
   *submissions* were rejected for 2021, yet the 2021 Return A file contains 12,459+ local
   agencies — converted NIBRS. Native SRS acceptance resumed in 2022.
4. **"Agencies appear in either the SRS file or the NIBRS file."** NIBRS reporters appear
   in *both* (the FBI back-converts). Only the ~2k native-SRS holdouts are SRS-only.
5. **"The hierarchy rule is the main difference between SRS and NIBRS counts."** It is a
   ~2.4% effect at the Part I level. Per-victim counting (person crimes), per-vehicle
   counting (MVT), the hotel rule, and the revised rape definition are the 5–25% effects.
6. **"NIBRS rape = the 'rape' offense code."** UCR-definition rape = 11A+11B+11C. 11A
   alone is ~25% low.
7. **"A correct re-derivation should match the published tables exactly."** Only at
   matched vintage. Master files are revised after publications freeze; ~1/3 of 2022
   disagreeing cells match neither current source. Within ~1% at mismatched vintage is
   success, not failure.
8. **"The CDE estimates file is the sum of the agency data."** It includes FBI estimation
   (~4–6% nationally in normal years, ~26–28% in 2021). Benchmark against it; never
   subtract from it to define a local remainder.

---

## 8. Repo terminology map and known misnomers

| Repo term | Meaning | Caveat |
|---|---|---|
| `srs_return_a_annual` | Annual Part I counts from the Kaplan Return A master file | Name is accurate (it *is* the Return A file) but remember §3: post-2020 these rows are mostly FBI-converted NIBRS, not native SRS submissions. |
| `nibrs_srs_equivalent_annual` | This repo's own rollup of the NIBRS offense + victim + property segments using the FBI's SRS scoring rules (§4) | Verified within ~1% of converted Return A on 2023–2024 dual agencies. Used as preferred source only for NIBRS-only agencies (~1.3% of count volume). Replaced the biased incident-distinct rollup (`nibrs_incident_annual`) in June 2026. |
| `cius_publication_annual` | Parsed CIUS/RCN "Offenses Known" local agency rows (Tables 8/9/11) | Frozen publication snapshot (§5). |
| `conversion_status = "as_released_return_a"` | "Value as released in the Return A master file." Deliberately *not* "native SRS" — for most recent-year agencies the row is FBI-converted NIBRS. (Renamed from the misnomer `native_summary` in June 2026.) | To actually distinguish converted vs native rows, test NIBRS batch-header membership for that ORI-year; the Return A file carries no flag. |
| `months_reported` (SRS rows) | True usable reported months, derived from monthly `number_of_months_missing` | FBI `last_month_reported` conventions differ; see SPEC. For converted-NIBRS rows this reflects months with NIBRS submissions. |
| Source priority CIUS > local pub > state pub > SRS > NIBRS | Publication-anchored preference | See §5 for what this choice means in 2020–2022. |

Directory names under `data/` follow the upstream source's own naming
(`SRS-Kaplan-1960-2024`, `NIBRS-Kaplan-1991-2024`, `FBI-CIUS-Annual`,
`FBI-CDE-Estimates-1979-2024`) and are retained as-is deliberately.

---

## 9. Reproduction and sources of record

* **Re-verifying the numbers in this document** is a development-workspace activity: in a
  full checkout with the raw data staged under `data/`,
  `uv run python scripts/diagnostics/verify_fbi_conversion_rules.py` regenerates every
  figure (writing `state/qa/fbi_conversion_rules_verification.md`). These scripts and their
  intermediate artifacts are not part of the clean submission package.
* **FBI**: "UCR Summary of Reported Crimes in the Nation, 2024" and "Reported Crimes in
  the Nation, 2024 FAQs" (cde.ucr.cjis.gov) — NIBRS-to-SRS conversion statement,
  submission-mix numbers (14,601 / 2,074 / 87.2% / 95.6%), estimation methodology.
* **FBI UCR Handbook** (ucr.fbi.gov/additional-ucr-publications/ucr_handbook.pdf) — the
  published scoring rules behind §4: per-victim scoring for crimes against persons,
  per-distinct-operation for property crimes, per-vehicle for MVT, hotel rule, hierarchy
  rule. "Reporting Rape in 2013" (ucr.fbi.gov) — the revised rape definition (11A+11B+11C).
* **Kaplan**: `docs/ucrbook_full.md` — conversion and 2021 history (~lines 148–162,
  510–520, 800–812, 1800–1812).
* **AGS methodology vintages**: `docs/AGS-CrimeRisk-Methodology-2025B.txt` and
  `docs/AGS-CrimeRisk-Methodology-2026A.pdf`. Material 2026A changes: projection
  geography to census block; "compiles … via NIBRS" wording added; "only the largest
  cities" → "many cities"; incident-data city list opened up (names New York, Los
  Angeles, Chicago); modeling jurisdictions ~10,000 → ~13,000; claimed model fit
  "over 85%" → "over 75%" of variance. The data source line ("FBI UCR, 2018–2024") is
  unchanged. AGS prose is imprecise about the mechanics; this guide, not the AGS text,
  governs how we describe the FBI layer.
