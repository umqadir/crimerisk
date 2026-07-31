# Repo Retention Inventory

This note lists the non-raw, non-code materials intentionally retained in the main repository
and explains exactly how each category is used.

## Repository Documentation And References

- `README.md`
  The product-facing repository overview. Documentation only. Not read by the build.

- `docs/SPEC.md`
  The technical build specification for the working repo. Documentation only. Not read by the
  build.

- `docs/TRACKER.md`
  The running tracker and work log retained in the working repo. Documentation only. Not read by
  the build.

- `docs/AGS-CrimeRisk-Methodology-2025B.txt` and `docs/AGS-CrimeRisk-Methodology-2026A.pdf`
  External AGS methodology references retained for citation and comparison (2026A is the current
  comparator; 2025B is kept as background). Not read by the live build.

- `scripts/release/assets/Manual_Input_Inventory.md` and
  `scripts/release/assets/manual_input_inventory.csv`
  The submission-package inventory sources. Documentation only. Not read by the build.

- `scripts/release/assets/Repo_Retention_Inventory.md` and
  `scripts/release/assets/repo_retention_inventory.csv`
  This repo-retention inventory and its machine-readable companion. Documentation only. Not read
  by the build.

## Live Config Files Read Directly By The Build

- `configs/agency_master_supplement.csv`
  Hand-maintained supplement for the agency master. `build-agency-master` and the reference build
  read it directly.

- `configs/local_resolution_overrides.csv`
  Hand-entered overrides for agency-to-place or agency-to-jurisdiction cases where the automated
  local resolver is wrong or ambiguous. `build-reference-layers` reads it directly.

- `configs/municipal_geometry_overrides.csv`
  Hand-entered geometry fixes for municipal support problems discovered in QA or map review.
  Geometry/allocation code reads it directly.

- `configs/overlap_custom_footprints.csv`
  Hand-curated custom service footprints for overlapping or special-purpose agencies whose
  geography cannot be inferred generically. Geometry/allocation code reads it directly.

- `configs/overlap_footprint_overrides.csv`
  Hand-selected choices among competing overlap-footprint candidates. Geometry/allocation code
  reads it directly.

- `configs/reporting_regime_overrides.csv`
  Hand corrections for agency-year reporting-regime cases where the automated classifier produced
  an implausible result. `build-reporting-regimes` reads it directly.

- `configs/source_preference_overrides.csv`
  Hand corrections for agency-year-offense source arbitration when the automated ranking is known
  to be wrong. Observations and controls read it directly.

- `configs/city_incident_sources.csv`
  Curated registry of benchmark-city incident sources, packet keys, and recommended disposition.
  `build-city-incident-shares` reads it directly.

- `configs/city_incident_priority.csv`
  Curated city-priority list. `build-city-incident-shares` reads it directly to order candidate
  city lanes before applying the promoted packet status gates.

- `configs/city_incident_categories/*.csv`
  Hand-authored offense-category mappings for cities whose labels cannot be handled by generic
  rules. The relevant city share builders read these files directly.

## Retained But Not Wired Live

- `configs/templates/*.csv`
  Dormant starter templates retained as examples for future override work. They are not wired into
  the live build.

## Reference-Matching Review Material

This workstream has three layers. First, the review worksheets under
`state/review/queues/local_resolution/` are where difficult agency-jurisdiction cases are
examined and resolved. Second, `promote-reference-inputs` takes the specific reviewed queue
outputs that matter for the live build and copies them into the canonical promoted input root
under `state/reference/inputs/`. Third, `build-reference-layers` reads only those promoted
parquets, plus the direct config overrides, to build the final jurisdiction universe and
crosswalk.

- `state/review/queues/local_resolution/local_agency_manual_review.*`,
  `agency_jurisdiction_review.*`, and `nonmunicipal_manual_review.*`
  Review worksheets and candidate tables used to reach human decisions about difficult
  agency-jurisdiction matches. They are not read directly by the live build and are not read by
  `promote-reference-inputs`. They are retained so the final decisions can be audited.

- `state/review/queues/local_resolution/provisional_local_agency_matches.*`
  First-pass local match surface produced by the local-resolution workflow.
  `promote-reference-inputs` reads the parquet here and copies it to
  `state/reference/inputs/provisional_local_agency_matches.parquet`.

- `state/review/queues/local_resolution/local_queue_resolved_final.*`
  Final reviewed local-tail resolution table.
  `promote-reference-inputs` reads the parquet here and copies it to
  `state/reference/inputs/local_queue_resolved_final.parquet`.

- `state/review/queues/local_resolution/nonmunicipal_special_resolved_final.*`
  Final reviewed table for nonmunicipal special cases and overlap-sensitive cases.
  `promote-reference-inputs` reads the parquet here and copies it to
  `state/reference/inputs/nonmunicipal_special_resolved_final.parquet`.

- `state/review/queues/local_resolution/nonmunicipal_auto_defaults.*`
  Deterministic nonlocal default table produced by the review workflow.
  `promote-reference-inputs` reads the parquet here and copies it to
  `state/reference/inputs/nonmunicipal_auto_defaults.parquet`.

- `state/reference/inputs/provisional_local_agency_matches.parquet`
  Canonical promoted reference input. `build-reference-layers` reads it directly.

- `state/reference/inputs/local_queue_resolved_final.parquet`
  Canonical promoted reference input containing the reviewed local-tail decisions.
  `build-reference-layers` reads it directly.

- `state/reference/inputs/nonmunicipal_special_resolved_final.parquet`
  Canonical promoted reference input containing the reviewed nonmunicipal special-case and
  overlap-sensitive decisions. `build-reference-layers` reads it directly.

- `state/reference/inputs/nonmunicipal_auto_defaults.parquet`
  Canonical promoted reference input containing the deterministic nonlocal defaults retained when
  manual escalation is unnecessary. `build-reference-layers` reads it directly.

## Municipal Publication Review Material

This workstream also has a staged flow. The packet directory under
`state/review/packets/municipal_targets/` is the research and review workspace for each
municipality. `promote-local-publications` then copies only the production-ready packet files into
the canonical promoted packet root under `state/modeling/inputs/local_publication/`. From there,
the local-publication builder consolidates the promoted packet files into
`state/modeling/inputs/local_publication_annual.parquet`, and that consolidated parquet is what
the observations and controls stages actually read.

- `state/review/packets/municipal_targets/*/packet_manifest.json`
  Packet metadata for a municipal-publication case. Retained as provenance. Not read directly by
  downstream live-build stages.

- `state/review/packets/municipal_targets/*/sources.csv`
  Source inventory for a municipal-publication case. Retained as provenance. Not read directly by
  downstream live-build stages.

- `state/review/packets/municipal_targets/*/research_findings.json`
  Review narrative for a municipal-publication case. Retained as provenance. Not read directly by
  downstream live-build stages.

- `state/review/packets/municipal_targets/*/recommendation.csv`
  Final packet-level production recommendation.
  `promote-local-publications` copies the production-ready packet files, including this file, into
  `state/modeling/inputs/local_publication/<case_key>/`.

- `state/review/packets/municipal_targets/*/published_reference_extract.csv`
  Structured publication extract in the generic packet format.
  `promote-local-publications` copies it into
  `state/modeling/inputs/local_publication/<case_key>/`, and the local-publication builder then
  reads the promoted copy.

- `state/review/packets/municipal_targets/*/*_offense_extract.csv`
  Structured publication extract in a packet-specific offense format such as the Quincy extract.
  `promote-local-publications` copies it into
  `state/modeling/inputs/local_publication/<case_key>/`, and the local-publication builder then
  reads the promoted copy.

- `state/modeling/inputs/local_publication/*/packet_manifest.json`
  Canonical promoted municipal packet metadata. The local-publication builder reads it directly.

- `state/modeling/inputs/local_publication/*/recommendation.csv`
  Canonical promoted municipal packet recommendation. The local-publication builder reads it
  directly to decide whether the packet is production-ready and which jurisdiction/ORI context to
  use.

- `state/modeling/inputs/local_publication/*/published_reference_extract.csv`
  Canonical promoted municipal publication extract in the generic packet format. The
  local-publication builder reads it directly when constructing
  `state/modeling/inputs/local_publication_annual.parquet`.

- `state/modeling/inputs/local_publication/*/*_offense_extract.csv`
  Canonical promoted municipal publication extract in a packet-specific offense format. The
  local-publication builder reads it directly when constructing
  `state/modeling/inputs/local_publication_annual.parquet`.

- `state/modeling/inputs/local_publication_annual.parquet`
  Consolidated canonical annual local-publication surface built from the promoted packet root.
  Observations and controls read this parquet directly.

## City Incident Review Material

The city-incident lane follows the same general pattern. The review packet tree under
`state/review/packets/city/` is where source choice, offense harmonization, and geography checks
are recorded. `promote-city-incident-inputs` copies the operational packet files into the
canonical promoted root under `state/modeling/inputs/city_incident/`. The city-share build then
reads only those promoted files, together with the direct city config files in `configs/`.

- `state/review/packets/city/*/packet_manifest.json`,
  `packet_status.csv`, `packet_checklist.csv`, `source_candidate.csv`,
  `research_findings.json`, and `reconciliation_summary.csv`
  Packet-level city review materials created during benchmark-city source research, offense
  harmonization, and geography validation. They are not read directly by the city-share build.

- `state/modeling/inputs/city_incident/*/packet_status.csv`
  Canonical promoted city packet status file.
  `promote-city-incident-inputs` copies it from the review packet tree, and
  `build-city-incident-shares` reads it directly to decide whether a city is production-ready.

- `state/modeling/inputs/city_incident/*/packet_offense_status.csv`
  Canonical promoted offense-level gate file.
  `build-city-incident-shares` reads it directly to decide which offenses are active for that
  city.

- `state/modeling/inputs/city_incident/*/offense_crosswalk.csv`
  Canonical promoted crosswalk from city-native incident labels to the project offense ontology.
  The relevant city share builders read it directly.

- `state/modeling/inputs/city_incident/*/published_reference_extract.csv`
  Canonical promoted reference extract for a city packet. Some city share builders read it
  directly for city-specific geography/reference context.

## State Publication Research Material

These state-source packet files are mostly explanatory rather than operational. They capture how
the state-publication lanes were researched, compared, and scoped. The live build does not read
most of them directly. Instead, the implemented loader code in `src/crimerisk/` produces
`state/modeling/inputs/state_publication_annual.parquet`, and the observations and controls stages
read that canonical parquet.

- `state/review/packets/source/states/*/packet_manifest.json`,
  `packet_status.csv`, `research_findings.json`, `source_recommendations.csv`,
  `state_priority_rows.csv`, `issue_bucket_summary.csv`, and `supporting_issue_metrics.csv`
  State-source research artifacts retained as provenance. They are not read directly by the live
  build.

- `state/review/packets/source/states/fl/fdle_fibrs_2024_ingestion_contract.json`,
  `fdle_fibrs_2024_workbook_schema.json`, and
  `state/review/packets/source/states/ms/official_ingestion_contract.json`
  Documentation/provenance describing how the Florida and Mississippi official-publication loaders
  were designed. The live build does not read these JSON files directly; the implemented loader
  logic lives in code and writes `state/modeling/inputs/state_publication_annual.parquet`.

- `state/review/packets/source/states/ca/reference_master_omission_patch.csv`,
  `state/review/packets/source/states/tx/reference_master_omission_patch.csv`, and
  `state/review/packets/source/reference_master_omission_patch_all.csv`
  Curated reference-gap patch lists retained as provenance from state source review. Not wired
  into the normal live build.

## Review Support And Provenance

- `state/review/support/service_structure_notes.csv`
  Service-structure notes collected during local-resolution review. Provenance only. Not wired.

- `state/review/support/methodology_audit_log.csv`
  Running methodology-change log. Provenance only. Not wired.

- `state/review/support/model_diagnostic_review.csv`
  Notes summarizing model experiments and benchmark interpretation. Provenance only. Not wired.

- `state/review/support/feature_source_inventory.csv`
  Inventory of candidate feature families and source status. Provenance only. Not wired.

- `state/review/support/seeds/*.csv`
  Seed lists used to start review queues and packet scaffolds. Support material only. Not wired.

- `state/review/runs/**/*.json`,
  `state/review/packets/source/states/*/codex_worker_*.txt`, and
  `state/review/packets/source/states/*/codex_worker_*.jsonl`
  Run logs and worker outputs from agent-assisted review sweeps. Provenance only. Not wired.

## Not Itemized Here

- `data/**`
  Raw and prepared source data.

- `src/**`, `main.py`, and most of `scripts/**`
  Executable code.

- `state/reference/*.parquet` outside `state/reference/inputs/`, `state/observations/**`,
  `state/controls/**`, `state/geometry/**`, `state/output/**`, and most of `state/modeling/**`
  Mechanically derived build outputs rather than retained manual-input material.

- `state/review/analysis/**`
  Generated diagnostics and analysis outputs.

- `state/review/queues/**` outside the explicitly named local-resolution files above
  Generated candidate queues or audit tables rather than retained manual-input surfaces.

- `archive/**`
  Historical material retained for provenance, not part of the live build surface.
