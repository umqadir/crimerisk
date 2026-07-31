# Manual Input Inventory

This note lists the non-raw, non-code materials retained in the submission package and explains
exactly how each category is used. Paths are shown as they appear inside the package.

## Release Documentation And References

- `README.md`
  The package overview. It is documentation only and is not read by the build.

- `report/CrimeRisk_Submission_Report.md`
  The generated canonical narrative report. It is documentation only and is not read by the build.

- `report/Manual_Input_Inventory.md`
  This inventory. It is documentation only and is not read by the build.

- `references/AGS-CrimeRisk-Methodology-2025B.txt`
  External AGS methodology reference included for citation and comparison. It is not read by the
  build.

## Live Config Files Read Directly By The Build

- `repro/configs/agency_master_supplement.csv`
  Hand-maintained supplement for the agency master. It adds or corrects agency metadata that is
  not reliable in the raw FBI/state rosters alone. `build-agency-master` and the reference build
  read it directly.

- `repro/configs/local_resolution_overrides.csv`
  Hand-entered overrides for agency-to-place or agency-to-jurisdiction cases where the automated
  local resolver is known to be wrong or ambiguous. `build-reference-layers` reads it directly.

- `repro/configs/municipal_geometry_overrides.csv`
  Hand-entered fixes for municipal geometry support problems discovered in QA or map review.
  Geometry/allocation code reads it directly.

- `repro/configs/overlap_custom_footprints.csv`
  Hand-curated custom footprints for special overlapping agencies whose service area cannot be
  inferred generically. Geometry/allocation code reads it directly.

- `repro/configs/overlap_footprint_overrides.csv`
  Hand-selected choices among competing overlap-footprint candidates. Geometry/allocation code
  reads it directly.

- `repro/configs/reporting_regime_overrides.csv`
  Hand corrections for agency-year reporting-regime cases where the automated classifier produced
  an implausible result. `build-reporting-regimes` reads it directly.

- `repro/configs/source_preference_overrides.csv`
  Hand corrections for agency-year-offense source arbitration when the automated preference order
  is known to be wrong. Observations and controls logic read it directly.

- `repro/configs/city_incident_sources.csv`
  Curated registry of benchmark-city incident sources, packet keys, and recommended disposition.
  `build-city-incident-shares` reads it directly when deciding which city packet lanes are active.

- `repro/configs/city_incident_priority.csv`
  Curated city-priority list. `build-city-incident-shares` reads it directly to order candidate
  city lanes before filtering that list using the promoted city packet status files.

- `repro/configs/city_incident_categories/*.csv`
  Hand-authored offense-category mappings for cities whose native offense labels cannot be handled
  by generic rules. The relevant city share builders read these files directly.

## Reference-Matching Review Material

This material is used in three stages. First, the review worksheets under
`repro/state/review/queues/local_resolution/` are where difficult agency-jurisdiction cases are
examined and resolved. Second, `promote-reference-inputs` takes the specific reviewed queue
outputs that matter for the release build and copies them into the canonical promoted input root
under `repro/state/reference/inputs/` during the package rebuild. Third,
`build-reference-layers` reads only those promoted parquets, plus the direct config overrides, to
build the final jurisdiction universe and crosswalk. The promoted parquets are regenerated during
`build-release`; they are not retained in the package as manual-input artifacts.

- `repro/state/review/queues/local_resolution/local_agency_manual_review.*`,
  `agency_jurisdiction_review.*`, and `nonmunicipal_manual_review.*`
  These are review worksheets and candidate tables used to reach human decisions about difficult
  agency-jurisdiction matches. They are not read directly by the live build and they are not the
  canonical reference inputs. They exist so a reviewer can see how the final decisions were made.

- `repro/state/review/queues/local_resolution/provisional_local_agency_matches.*`
  This is the first-pass local match surface produced by the local-resolution workflow. During a
  release rebuild, `python main.py promote-reference-inputs` reads the parquet version from this
  review-queue path and copies it to
  `repro/state/reference/inputs/provisional_local_agency_matches.parquet`.

- `repro/state/review/queues/local_resolution/local_queue_resolved_final.*`
  This is the final reviewed local-tail resolution table. It is not read directly by
  `build-reference-layers`. Instead, `promote-reference-inputs` reads the parquet here and copies
  it to `repro/state/reference/inputs/local_queue_resolved_final.parquet`.

- `repro/state/review/queues/local_resolution/nonmunicipal_special_resolved_final.*`
  This is the final reviewed table for nonmunicipal special cases and overlap-sensitive cases.
  `promote-reference-inputs` reads the parquet here and copies it to
  `repro/state/reference/inputs/nonmunicipal_special_resolved_final.parquet`.

- `repro/state/review/queues/local_resolution/nonmunicipal_auto_defaults.*`
  This is the deterministic nonlocal default table produced by the review workflow. The release
  rebuild does not read it directly from the queue path. `promote-reference-inputs` copies the
  parquet to `repro/state/reference/inputs/nonmunicipal_auto_defaults.parquet`.

## Municipal Publication Review Material

This material also has a staged flow. The packet directory under
`repro/state/review/packets/municipal_targets/` is the research and review workspace for each
municipality. `promote-local-publications` then copies only the production-ready packet files into
the canonical promoted packet root under `repro/state/modeling/inputs/local_publication/`. From
there, the local-publication builder consolidates the promoted packet files into
`repro/state/modeling/inputs/local_publication_annual.parquet`, and that consolidated parquet is
what the observations and controls stages actually read. Those promoted packet copies and the
consolidated annual parquet are rebuilt during `build-release`; they are not retained in the
package as manual-input artifacts.

- `repro/state/review/packets/municipal_targets/*/recommendation.csv`,
  `packet_manifest.json`, `published_reference_extract.csv`, and `*_offense_extract.csv`
  These are the packet files that matter operationally. `python main.py promote-local-publications`
  syncs the production-ready packet files into
  `repro/state/modeling/inputs/local_publication/<case_key>/`.

## City Incident Review Material

The city-incident lane follows the same general pattern. The review packet tree under
`repro/state/review/packets/city/` is where source choice, offense harmonization, and geography
checks are recorded. `promote-city-incident-inputs` copies the operational packet files into the
canonical promoted root under `repro/state/modeling/inputs/city_incident/`. The city-share build
then reads only those promoted files, together with the direct city config files in
`repro/configs/`. Those promoted copies are rebuilt during `build-release`; they are not retained
in the package as manual-input artifacts.

- `repro/state/review/packets/city/*/packet_status.csv`,
  `packet_offense_status.csv`, `offense_crosswalk.csv`,
  `published_reference_extract.csv`, and `reconciliation_summary.csv`
  These are the reviewed gating and reconciliation files that
  `python main.py promote-city-incident-inputs` syncs into
  `repro/state/modeling/inputs/city_incident/<city_key>/` before the city-share build runs.

## Package Boundary For Review Material

This package includes only the reviewed-input state that the canonical rebuild actually consumes:

- `repro/state/review/queues/local_resolution/**`
- `repro/state/review/packets/municipal_targets/` limited to the production-consumed packet files
- `repro/state/review/packets/city/` limited to the production-consumed packet files

Broader review provenance such as state-source research packets, support notes, and batch-run logs
remain in the working repository rather than the submission package. They explain how some lanes
were researched, but they are not required by `uv run python main.py build-release --emit-fbi-calibrated`
or by the standard QA path in this package.

## Not Itemized Here

- `repro/data/**`
  Raw and prepared source data. These files are intentionally not bundled in the clean submission
  package; the package documents the expected paths for anyone rebuilding from supplied data.

- `repro/src/**`, `repro/main.py`, `repro/scripts/**`, `repro/pyproject.toml`,
  `repro/uv.lock`, and `repro/.python-version`
  Executable code and environment metadata.

- regenerated promoted inputs and derived state such as:
  `repro/state/reference/**`, `repro/state/observations/**`, `repro/state/controls/**`,
  `repro/state/geometry/**`, `repro/state/output/**`, and most of `repro/state/modeling/**`
  Mechanically derived build outputs rather than manual-input material.

- `repro/state/review/analysis/**`
  Generated diagnostics and analysis outputs.

- `repro/state/review/queues/**` outside the explicitly named local-resolution files above
  Generated candidate queues or audit tables rather than retained manual-input surfaces.

- `materials/**` and `validation/**`
  Derived analytical deliverables rather than manual-input material.
