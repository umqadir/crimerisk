# CrimeRisk Methodology

CrimeRisk is a national, cross-sectional estimate of neighborhood-level crime risk for the
United States, built entirely from public data. This document describes what the estimate
is, how it is produced, how it is tested, and where it should not be trusted. It is written
for a reader deciding whether the surface is credible enough to use: a researcher, a data
journalist, or someone evaluating it against a commercial alternative. Every quantitative
claim below is reproducible from the released outputs and the configuration files that drive
the build.

## What the surface represents

The product is a single reference year, 2024, estimated at two nested geographies: the
census block group (a small statistical area, typically 600 to 3,000 residents) and the
census tract (a few block groups). The block-group surface covers 238,193 block groups; the
tract surface covers 83,776 tracts. These are the populated block groups and tracts of the
fifty states and the District of Columbia.

For each geography the surface reports three kinds of quantity, all for the seven "Part I"
offenses that the FBI's Uniform Crime Reporting program has tracked for decades (murder,
rape, robbery, aggravated assault, burglary, larceny-theft, and motor vehicle theft):

- An expected count: the modeled number of each offense attributable to that block group in
  2024. Summed across the country, expected counts total about 7.17 million Part I offenses.
  The block-group counts and the tract counts sum to the same national total, and each block
  group's counts sum to its jurisdiction's official total. This internal consistency is
  enforced, not assumed.
- A per-capita index for each offense, scaled so that 100 equals the national rate. An index
  of 250 for burglary means the block group's modeled burglary rate is two and a half times
  the national average; an index of 40 means well below it. The index is a relative measure,
  not a probability.
- A density layer: expected offenses per square mile of land area, which does not depend on
  any population or exposure denominator and is therefore defined even where per-capita rates
  are not (for example, in a retail district with almost no residents).

Several aggregate indices combine the seven offenses: a total, a personal-crime subtotal
(murder, rape, robbery, aggravated assault), a property subtotal (burglary, larceny, motor
vehicle theft), and a harm-weighted total that gives more weight to more serious offenses.

Two things the surface is deliberately not. It is not a prediction about any individual, home,
or address: it is a small-area rate, and the risk to a specific person depends on
circumstances the model does not see. And it is not incident data. It contains no records of
individual crimes, no locations of specific events, and no personally identifying information.
It is a modeled statistical surface built by allocating official aggregate counts across
neighborhoods.

## The two-lane design

The central design decision is a strict separation between how much crime a jurisdiction has
and how that crime is distributed inside the jurisdiction. These are handled by two independent
lanes that meet only at the end.

The totals lane sets levels. For every law-enforcement jurisdiction, an official 2024 annual
count is established for each offense from published sources. Nothing in the neighborhood
modeling can change these totals.

The share lane sets shape. Within each jurisdiction, a distribution across block groups says
what fraction of the jurisdiction's total falls in each neighborhood. This distribution comes
from geocoded incident data where good incident data exist, and from a statistical model of
neighborhood characteristics everywhere else.

The final estimate for a block group is simply its jurisdiction's official total multiplied by
its within-jurisdiction share, computed offense by offense. Because the shares within a
jurisdiction are constrained to sum to one, the block-group counts automatically reconcile to
the official total, a step called raking. Every published rate and index is then derived from
these expected counts by dividing by the appropriate denominator. There is no separate display
layer, no post-hoc smoothing of the published map, and no path by which a rate is written
independently of its count. This is the property that keeps the counts and the indices
mutually consistent.

This structure mirrors the approach used by established commercial crime-risk products: an
authoritative external total, distributed within jurisdictions by a neighborhood model
informed by local incident data. CrimeRisk keeps that principle and adds explicit,
machine-readable metadata about which lane produced each value and how reliable it is.

## The totals lane: official jurisdiction counts

The counts that set jurisdiction levels come from the FBI's Uniform Crime Reporting data,
supplemented by state publications where a state's own system is more complete than the
federal one.

Some background on the federal data matters here, because it is widely described incorrectly.
The FBI now collects most crime data through the National Incident-Based Reporting System
(NIBRS), an incident-level format, and converts it into the older Summary Reporting System
(Return A) counts that the annual publications are built from. For 2024, of the local police
agencies in the reference panel, about 86 percent report through NIBRS and are back-converted
by the FBI; the remainder still submit summary counts directly. This means that anyone using
the FBI's published agency tables, including this product, is consuming FBI-converted NIBRS
for the large majority of agencies. The conversion is not a detail: counting rules differ by
offense (crimes against persons are counted once per victim, motor vehicle theft once per
stolen vehicle, and the 2013 revised rape definition sums three incident codes rather than
one), and getting them wrong understates individual offenses by as much as 25 percent. The
repository implements and verifies each of these rules against the FBI's own converted totals;
on 2024 agencies that report both ways, the reconstruction matches the FBI's converted counts
in 98.8 percent of agency-offense cells.

Source selection follows a fixed priority: the FBI's frozen annual publication first, then a
local agency's own published annual figure, then a state publication, then the summary master
file, then the repository's own NIBRS-to-summary rollup for agencies the other sources miss.
The publication is preferred over the living master file because the published record is the
auditable public number; for the 2023 and 2024 reference years the two agree in 99.9 percent
of comparable cells in any case.

Two states are handled through their own publications rather than the federal tables. Florida
does not participate in NIBRS in a way that produces complete federal counts, so Florida's
totals come from the Florida Department of Law Enforcement's published figures. New York's
totals come from the state Division of Criminal Justice Services, which publishes agency and
county totals (including estimates for agencies that report partially to the FBI) and which
resolved a specific, verified defect in one large county's federal submission. In both cases
the state is treated uniformly: one source contract per state, applied to every agency, rather
than a per-agency patchwork.

## The share lane: distributing totals within jurisdictions

Within a jurisdiction, the share of each offense that falls in each block group is estimated
by blending two sources of evidence.

The first is direct incident texture: geocoded incident records from a city's own open-data
feed, aggregated to block groups. Thirty-five cities and urban counties have incident data of
sufficient quality and completeness to serve as within-city allocation evidence: Atlanta,
Aurora (CO), Austin, Baltimore, Baton Rouge, Boston, Charlotte, Chicago, Cincinnati,
Cleveland, Colorado Springs, Dallas, Denver, Durham, Fort Worth, Houston, Indianapolis,
Jacksonville, Kansas City (MO), Memphis, Mesa, Milwaukee, Minneapolis, Montgomery County (MD),
New York, Oakland, Omaha, Philadelphia, Sacramento, San Diego, San Francisco, Seattle,
St. Louis, Tucson, and Washington, DC. Together these jurisdictions contain roughly 27,000 of
the country's 238,193 block groups, on the order of one in nine. For every other block group,
there is no local incident feed.

The second source is a national covariate model: a gradient-boosted regression (a
tree-ensemble method) trained to predict where offenses concentrate within a jurisdiction from
a classified pool of 346 block-group characteristics. These features describe commercial and
retail activity, land cover and land use, road and transit structure, population density,
housing, and socioeconomic composition drawn from the American Community Survey, employment
data from the Census Bureau's LODES program, points of interest from Overture, land cover from
the National Land Cover Database, and road and transit geography. The within-jurisdiction
allocation model draws 296 candidate features from the permitted classes of that pool after a
policy screen described below.

Where a city has usable incident data, the two sources are combined in a posterior: the
observed incident distribution is treated as evidence and blended with the model's prediction,
with the weight on the incident data determined by how well the feed reconciles to the
jurisdiction's official total. A feed whose annual counts closely match the official total is
trusted heavily and dominates the block-group shares; a feed that under-reports, over-reports,
or is sparse is pulled back toward the model. Where a city has no incident data (most of the
country), the shares come from the model alone.

How far the covariate model is trusted in uncovered areas depends on the offense, and this is
set by held-out evidence rather than by preference. Robbery, aggravated assault, larceny, and
motor vehicle theft use the full covariate signal, because the model demonstrably beats a
simple population or exposure baseline out of sample for these offenses. Burglary uses a
calibrated partial transfer, selected by a one-standard-error parsimony rule on cross-validated
error. Murder and rape use only a population or exposure baseline in uncovered areas: their
within-jurisdiction covariate signal does not generalize out of sample, so the product does not
pretend to model where murders or rapes concentrate in cities it has no data for. This is a
real limitation, stated as one rather than hidden.

A bounded, distance-decaying adjustment carries a covered city's signal a short way into its
immediate surroundings and is then re-raked, so that a coverage boundary does not appear as an
abrupt seam without letting a city rewrite its rural neighbors.

### The no-redlining screen

Because the neighborhood model uses socioeconomic features, there is a real risk of encoding
demographic composition as if it were crime risk. The build addresses this with an explicit
feature policy. Direct demographic composition features (race, ethnicity, ancestry, language,
nativity, sex composition, family structure, and national-origin proxies) are excluded from
the within-neighborhood allocation model outright, regardless of predictive power. The policy
is derived from a test that separates two things a feature can do: help distinguish high-crime
from low-crime jurisdictions (a between-jurisdiction effect) versus help locate crime within a
jurisdiction (a within-jurisdiction effect). A feature that helps only between jurisdictions is
treated as ecological and kept out of neighborhood allocation. The current classification
assigns the pool of 346 features as follows: 27 direct demographic features are excluded
outright, 91 are between-jurisdiction-only and excluded from neighborhood allocation, one is
dropped as unstable, 158 are usable on both axes, and 69 socioeconomic proxies are retained
only under an explicit "proxy review" label, documented as redlining-adjacent rather than
treated as neutral. The build fails if any excluded or unmapped feature is found in the fitted
allocation model. This is a necessary safeguard, not a sufficient one: excluding
direct demographic fields does not remove all correlation with them, and the product does not
claim otherwise.

## Handling defective inputs

Public crime data are riddled with reporting artifacts, and a naive pipeline would convert
those artifacts into false neighborhood hotspots or phantom safe zones. A substantial part of
the build is dedicated to detecting these input pathologies and defaulting to the conservative
choice (falling back to the model or to a coarser total) rather than trusting a suspect
signal. Each mechanism below exists because a specific, verified failure was found in real
data.

Reporting-gap fills. Some agencies, particularly county sheriffs, report to the FBI for only
part of a year or not at all, which would make their jurisdictions appear nearly crime-free. A
per-agency fill estimates the missing counts from the agency's own reporting history rather
than letting the gap propagate. In one verified case, restoring a large county sheriff's
missing months moved its county rate from about 170 to about 848 offenses per 100,000
residents and its typical neighborhood index from near zero to a plausible level.

Masked-gap detection. More insidious than an obvious gap is an agency whose completeness flags
read as clean while its counts are materially incomplete: for instance, a year that reports
all twelve months but at half the agency's normal volume. A detector reclassifies these cases
into the same fill process. It currently flags on the order of fifteen agencies nationally;
correcting one state's largest city closed that state's total from about 14 percent below the
federal estimate to under 2 percent.

State-publication lanes. Where a state's own reporting system is demonstrably more complete
than the federal tables (Florida and New York), the state publication is used for the whole
state, as described above. This is a deliberate choice to prefer the more complete official
record, made at the level of a source contract rather than case by case.

Consolidated agency footprints. A few large agencies police an area much larger than their
nominal city. Two are handled explicitly: the Las Vegas Metropolitan Police Department, whose
service area covers much of Clark County beyond Las Vegas city, and Louisville Metro Police,
covering Jefferson County. For these, the agency's total is allocated over its true service
footprint (the principal city plus the county remainder) while separately reporting
municipalities inside the county are excluded so they are not double-counted.

Coordinate quarantine and reviewed exceptions. Incident feeds frequently place records at
default or masked coordinates: a police headquarters, a precinct centroid, a hospital where a
report was taken. Left alone, these create enormous false hotspots. A quarantine registry
removes named artifact coordinates from the incident evidence before aggregation (currently 108
entries), and a separate tripwire fails the release if any single point carries an implausible
share of a city's located incidents unless that point has been individually reviewed and
whitelisted as a genuine concentration: a mall, a transit hub, a large apartment complex (119
reviewed exceptions). In one city, quarantining a transit-station artifact dropped that block
group from 11 percent of the city's vehicle thefts to under a fifth of one percent.

Offense-level exclusions. When a feed's count for one offense cannot be reconciled and the
discrepancy is not explained by documented source omissions, that offense is excluded for that
city and its allocation falls back to the model, a fail-closed default. Two cases illustrate
the standard. A city police feed's aggravated-assault count ran about 40 percent below the
official total (roughly 7,450 against about 12,500) with no documented reason, and with the
strong possibility that the missing records were domestic-violence reports that are not
spatially neutral; that single offense was excluded while the city's other offenses were kept.
A second city's entire public feed was a structural undercount of roughly 62 percent from an
unaudited export filter, with homicides silently absent; that city was excluded entirely and
its feed is not used.

Rape texture defaults to deny. Rape incident locations are pervasively masked in public feeds:
coded to precinct station houses or to hospitals rather than to where the offense occurred. In
one large city, nearly 10,000 located rapes fell on exactly 77 precinct points. Because these
locations are systematically misleading, the product does not use direct rape point locations
for neighborhood texture by default; it allows them only for four cities whose rape geography
was individually audited and found diffuse and unmasked. Everywhere else, rape allocation falls
back to the baseline.

## How the surface is validated

Nothing reaches the published output except through a fail-closed release gate: a chain of
automated checks that must all pass, run against the actual rendered output files and map
tiles, not an intermediate. The current release passes every gate with zero open issues.

The gate has several blocks. Total-lane integrity confirms that jurisdiction controls reconcile
and that no incident feed has been promoted to a total without reconciling to the official
figure. Allocation coherence confirms that block-group counts sum to their controls, that tract
counts equal the sum of their block groups, that no count is negative, and that exactly one
share source is used per cell. Index coherence recomputes every published rate and index from
its expected count and the stored national normalizer and fails if any value was written
independently. This is the check that makes count-and-index incoherence impossible. Spatial-artifact
checks look for the failure modes that a crime map is prone to: state-border discontinuities,
tract-level flatness, checkerboard speckle, low-denominator hotspots, and coverage seams. A
feature-policy audit confirms the no-redlining exclusions hold in the fitted model.

The centerpiece of validation is held-out cross-validation on the within-city allocation. The
35 cities with incident data are used as a test set: the model is repeatedly retrained with one
city withheld, and its predicted neighborhood distribution for that city is scored against the
city's actual incident distribution. The scoring metric is total variation distance (TVD)
between the predicted and observed share distributions across the city's block groups. TVD
ranges from 0 (identical distributions) to 1 (no overlap); a TVD of 0.30 means that about 30
percent of the predicted incident mass would have to be moved between block groups to match
the observed distribution, and about 70 percent already overlaps. Because this is measured on
withheld cities, it estimates how the model performs where it has no local data, which is the
situation for most of the country.

The held-out, incident-weighted TVD for the current release, by offense:

| Offense | Held-out TVD | Cities scored |
|---|---|---|
| Motor vehicle theft | 0.29 | 33 |
| Larceny-theft | 0.31 | 34 |
| Aggravated assault | 0.33 | 28 |
| Burglary | 0.33 | 33 |
| Robbery | 0.41 | 33 |
| Rape | 0.64 | 4 |
| Murder | 0.72 | 27 |

The standard errors on these figures range from about 0.008 for the high-volume property
offenses to about 0.026 for murder. Each release must not degrade any offense against a frozen
baseline by more than one standard error; the current release improves robbery by about 2.6
standard errors, larceny by 1.8, and aggravated assault by 1.4 against that baseline, with no
offense degrading beyond noise. The rape figure rests on only four cities (the four with
audited, unmasked rape geography) and should be read as low-confidence.

A visual inspection stage renders the new surface and a set of unchanged control cities and
confirms that the intended geography appears and that untouched areas do not move. After the
surface is promoted, the full validator is run once more against the live output directory.

## Pre-registration

The most important guard against fooling ourselves is prospective. Before any new city's
incident data is examined, the model's predicted neighborhood distribution for that city (the
prediction it makes with no knowledge of the city's actual incidents) is frozen and committed,
along with the scoring metric and the expected error bands. The city's real data are then
obtained and scored against that frozen prediction. Because the prediction and the scoring
rule are fixed in advance, the result cannot be adjusted after the fact; any revision to the
frozen prediction after truth is seen is treated as a failed pre-registration rather than a
correction.

Two rounds of onboarding have run under this discipline. In the first, five cities were
pre-registered against the model's frozen predictions. Of 25 scoreable offense-tests across the
four cities that yielded usable data, 22 fell within or better than the pre-registered bands,
most of them better than predicted. The three misses were all murder in cities with very few
murders (35 to 61 incidents), where block-group distributions are inherently noisy; the one
onboarded city with a large murder count scored comfortably inside its band. In the second
round, Houston was pre-registered and then scored 7 of 7 offenses within or better than its
frozen bands (five better, two at the band edge), the strongest prospective result of any
onboarded city. Pre-registration converts each new city from a tuning opportunity into an
out-of-sample test of the error bars quoted above.

## What the accuracy numbers mean for a user

The honest reading of the validation is specific. For the high-volume offenses (larceny, motor
vehicle theft, aggravated assault, burglary), the model reproduces roughly two-thirds to
seventy percent of a city's neighborhood distribution in cities it has never seen. In practice
that means it reliably captures the broad within-city gradient: which parts of a city carry
more property crime and which carry less, and the general shape of the concentration. It does
not mean block-group precision. A single block group's index can be off substantially even
where the citywide pattern is right, and users should treat an individual block group's value
as an estimate with real uncertainty, not a measurement.

For robbery the model is somewhat weaker (TVD around 0.41), and for murder and rape it is weak
at the block-group level. Murder and rape are rare events; in most neighborhoods the expected
count is small, and small counts are volatile. The product does not repair this by overwriting
the estimates. Instead it publishes, for every cell, the direct incident support behind the
value, Poisson-based reliability intervals, a reliability tier, and a recommended coarser
display geography, and it suppresses per-capita rates entirely where the exposure denominator
is too small to support one. The intended use for murder and rape is at the tract or city
level and as a relative gradient, not as a block-group point estimate.

## Known limitations

The product's limitations are documented rather than smoothed over.

Rare-offense noise. As above, murder and rape carry substantial block-group noise. The
reliability metadata and a suppression floor for thin denominators are the honesty layer; the
recommendation is to view these offenses at coarser geography.

Modeled texture outside covered cities. For roughly nine in ten block groups there is no local
incident feed, and the within-jurisdiction distribution is produced by the covariate model
raked to the official total. The error of that modeling is measured only where city-like
incident truth exists: the 35 cities. Small-town and rural neighborhoods are extrapolation
beyond the validation support; one known symptom is that the cores of small county seats may
read as too muted. The distinction between directly-observed and modeled texture is carried in
the output metadata, offense by offense, so a user can see which they are looking at.

Temporal reference. The surface is for 2024. Jurisdiction totals are 2024; the within-city
incident shares pool several years to gain spatial stability and are then raked to 2024 levels.
Neighborhoods that changed sharply after the pooling window may carry a slightly stale shape.

Per-resident rates where few people live. The primary index for most offenses uses an exposure
denominator that combines residents with workplace and activity measures, because a downtown
with few residents is not infinitely dangerous per capita. Even so, places dominated by
visitors, tourists, or transit riders can overstate per-person risk, because no public dataset
measures the transient daytime population well. High-exposure cells carry a flag noting this,
and a density layer that needs no denominator is provided as an alternative view. A candidate
that adds modeled daytime population as a denominator floor has been evaluated but not adopted,
because it did not clearly improve the held-out result.

Dependence on official reporting quality. The jurisdiction totals are only as good as the
official data behind them. Where a state or agency under-reports to the sources used, the
product inherits that under-reporting as a lower level, even though the neighborhood shape may
be sound. Mississippi's low levels, for example, reflect genuine gaps in that state's official
reporting rather than a modeling choice, and Florida's totals rest on the state's own
publication by design.

Scope. The surface covers the fifty states and the District of Columbia; it is not a claim
about territories or special federal jurisdictions.
The product is an open, public-data estimate inspired by established commercial crime-risk
methodology, not a one-for-one replica of any commercial product, and no side-by-side
comparison against a commercial surface is claimed here.

## Reproducibility

The build is designed so that an outside party with the same inputs can reproduce the same
outputs.

Pinned sources with recorded provenance. Every external dataset is a specific frozen file with
a recorded producer chain and vintage: the FBI Return A and NIBRS master files (via Jacob
Kaplan's openICPSR concatenations, a fixed 2025 build), the FBI's published annual tables, the
Crime Data Explorer estimates, the American Community Survey five-year release, Census
population estimates, LODES employment data, Overture points of interest at a pinned release,
National Land Cover Database rasters, and TIGER geography. The federal conversion rules the
totals depend on are verified against the FBI's own converted counts and documented rule by
rule.

Deterministic, gated builds. A build writes a complete candidate output set to its own
directory together with a manifest recording the exact code commit, the resolved
configuration, input file identities, and output statistics. Promotion to the published output
is a separate, scripted step that verifies hashes and ancestry: the candidate's manifest, its
validation summary, the block-group data feeding the map tiles, and the rendered tiles are all
hash-checked so that a stale or mismatched artifact cannot be promoted. The published surface
records the source data file, the build commit, and the rendered field.

Configuration-driven onboarding. Adding a city is a configuration change, not a code change.
Each city's source, offense admissions, coordinate quarantines, and reviewed exceptions live in
committed configuration files, and a generic contract runner ingests any conforming feed
through the same path. The registries of admissions, quarantines, and exceptions are the
committed record of every human adjudication that shaped the surface.

## Evidence appendix

For readers working from a clone of the repository, the claims above map to these artifacts.
The block-group and tract output parquet files themselves are large generated data and are not
included in this repository; the items below are the curated evidence and configuration that
are included.

- Surface size, national totals, and per-offense published counts: the release validation
  summary (`state/output/validation_summary.json`) and the output build manifest
  (`state/output/crimerisk_output_build_2024.json`), which record the same figures
  (238,193 block groups, 83,776 tracts, national expected-count total 7,167,929) that the full
  `crimerisk_block_group_2024_ags_core.parquet` and `crimerisk_tract_2024_ags_core.parquet`
  outputs carry.
- Field and index definitions, denominators, and aggregate rules: the published-field policy
  and the output build manifest, which records every normalizer.
- Federal data mechanics, conversion rules, and the 98.8 percent reconciliation:
  `docs/FBI-DATA-GUIDE.md` and `scripts/diagnostics/verify_fbi_conversion_rules.py`.
- The 35 direct-evidence cities and their held-out scores: the nested cross-validation evidence
  (`state/modeling/nested_city_cv_2024.json` and the companion parquet).
- Covariate feature policy and the no-redlining exclusions: the feature-transfer policy artifact
  (`state/modeling/feature_transfer_policy_2024.parquet` and `.json`: 346 classified features:
  158 both-axes, 91 between-only, 69 proxy-review, 27 excluded-protected, 1 unstable) and the
  residual feature-policy block of the build manifest (296 candidate features).
- Data-hygiene mechanisms: the reporting-gap and masked-gap fills, the state-publication lanes,
  the consolidated-agency footprints (`configs/consolidated_agency_footprints.csv`), the
  coordinate quarantine (`configs/city_feed_coordinate_quarantine.csv`, 108 entries) and
  exact-point exceptions (`configs/city_feed_exact_point_exceptions.csv`, 119 entries), the
  offense-level admissions (`configs/gate17_city_offense_admissions.csv`), and the rape texture
  policy (`configs/city_offense_texture_policy.csv`).
- The release gate: `scripts/diagnostics/validate_release_outputs.py` and the promotion script's
  hash and ancestry checks.
- Pre-registration: the frozen prediction artifacts committed before each onboarding round
  (`state/modeling/batch_b_preregistration_2024.json` and
  `state/modeling/batch_c_preregistration_2024.json`, with their companion share parquets). The
  underlying per-city research and scoring notes are part of the internal development workspace
  and are not included in this repository.
