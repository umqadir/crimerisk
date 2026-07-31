# CrimeRisk Methodology

CrimeRisk is a cross-sectional, point-in-time estimate of police-recorded neighborhood crime
for 2024, built entirely from public data. It covers the forty-eight contiguous states and the
District of Columbia. This document describes what the estimate is, how it is produced, how it
is tested, and where it should not be trusted. Quantitative claims are reproducible from the
released outputs and build configuration.

## What the surface represents

The product is a single reference year, 2024, estimated at two nested geographies: the
census block group and the census tract. The block-group surface covers 238,193 block groups; the
tract surface covers 83,776 tracts. These are the populated block groups and tracts of the
forty-eight contiguous states and the District of Columbia; Alaska, Hawaii, and the
territories are out of scope on both sides of every comparison in this document.

For each geography the surface reports three kinds of quantity, all for the seven "Part I"
offenses that the FBI's Uniform Crime Reporting program has tracked for decades — murder,
rape, robbery, aggravated assault, burglary, larceny-theft, and motor vehicle theft:

- An expected count: the modeled number of each offense attributable to that block group in
  2024. On the mapped surface, expected counts total 7,178,829 Part I offenses.
  The block-group counts and the tract counts sum to the same national total, and block-group
  counts reconcile to their assigned jurisdiction or state-ledger component control. This
  internal consistency is enforced, not assumed.
- A primary index for each offense, scaled so that 100 equals the national reference rate
  under that offense's exposure denominator. An index of 250 for burglary means the modeled
  burglary rate is two and a half times its national reference; an index of 40 means well below
  it. The index is a relative measure, not a probability. For murder and rape, whose single-year
  counts are too sparse to support a rate at block-group scale, this per-offense index is
  published at the census tract and coarser
  rather than the block group; the block group retains their expected counts. The reasoning is
  set out under "What the accuracy numbers mean for a user."
- A density layer: expected offenses per square mile of land area, which does not depend on
  any population or exposure denominator and is therefore defined even where per-capita rates
  are not (for example, in a retail district with almost no residents). It is an incident
  intensity view, not the product's risk denominator.

Several aggregate indices combine the seven offenses — a total, a personal-crime subtotal
(murder, rape, robbery, aggravated assault), a property subtotal (burglary, larceny, motor
vehicle theft), and a harm-weighted total that gives more weight to more serious offenses.
Robbery, aggravated assault, burglary, larceny, motor vehicle theft, and the aggregates publish
at block-group scale. Murder and rape publish at tract scale; block-group aggregates retain
their tract-supported murder and rape contributions compositionally.

The surface is not a forecast: it describes the most recent complete reference year and is
refreshed annually. It is not a prediction about any
individual, home, or address: it is a small-area rate, and the risk to a specific person
depends on circumstances the model does not see. And it is not incident data. It contains no
records of individual crimes, no locations of specific events, and no personally identifying
information. It is a modeled statistical surface built by allocating official aggregate counts
across neighborhoods. It is inspired by established commercial methods, including AGS
CrimeRisk, but it does not claim one-for-one AGS parity.

The target quantity is the geography of *police-recorded* crime: both the signal the model
learns from and the truth it is validated against are police records. Crime that is never
reported to police, or never recorded by them, is in neither the training data nor the
validation data. A neighborhood where certain
offenses are systematically under-reported, or where incidents are poorly geocoded, therefore
inherits that bias in its estimate. Raking to official jurisdiction totals fixes the overall
level for a jurisdiction, but it cannot correct reporting or geocoding bias in how that total is
distributed across the neighborhoods within the jurisdiction.

## The two-lane design

The central design decision is a strict separation between how much crime a jurisdiction has
and how that crime is distributed inside the jurisdiction. These are handled by two independent
lanes that meet only at the end.

The totals lane sets levels. For every law-enforcement jurisdiction, an official 2024 annual
count is established for each offense from published sources — or, for the small share of
territory whose agencies reported nothing at all, a modeled sub-target capped by the FBI's own
published state estimate, described below and labelled as such in the output. Nothing in the
neighborhood modeling can change these totals.

The share lane sets shape. Within each jurisdiction footprint, a distribution across block
groups says what fraction of the jurisdiction's total falls in each neighborhood. This
distribution comes from geocoded incident data where good incident data exist, and from modeled
activity and exposure shares everywhere else. Outside the direct-incident cities, a published
block-group count is therefore an estimate of within-footprint allocation, not an observed
block-group incident count.

The final estimate for a block group is its assigned jurisdiction or state-ledger component
control multiplied by its within-footprint share, computed offense by offense. Because the
shares within a component are constrained to sum to one, the block-group counts automatically
reconcile to that control — a step called raking. Every published rate and index is then derived from
these expected counts by dividing by the appropriate denominator. There is no separate display
layer, no post-hoc smoothing of the published map, and no path by which a rate is written
independently of its count. This is the property that keeps the counts and the indices
mutually consistent.

This structure mirrors the approach used by established commercial crime-risk products: an
authoritative external total, distributed within jurisdictions by a neighborhood model
informed by local incident data. CrimeRisk keeps that principle and adds explicit,
machine-readable metadata about which lane produced each value and how reliable it is.

Jurisdiction estimates also reconcile through a state-control ledger. Municipal,
nonmunicipal, overlap, and benchmark components partition to one control for each state and
offense before publication; the companion FBI-calibrated tables reconcile to the published FBI
state estimates. State borders are not smoothing constraints. A step at a state line can reflect
different published FBI state rates, different state reporting systems, or different offense
definitions rather than a spatial write error.

## The totals lane: official jurisdiction counts

The counts that set jurisdiction levels come from the FBI's Uniform Crime Reporting data,
supplemented by state publications where a state's own system is more complete than the
federal one.

The FBI now collects most crime data through the National Incident-Based Reporting System
(NIBRS), an incident-level format, and converts it into the older Summary Reporting System
(Return A) counts that the annual publications are built from. For 2024, of the local police
agencies in the reference panel, about 86 percent report through NIBRS and are back-converted
by the FBI; the remainder still submit summary counts directly. This means that anyone using
the FBI's published agency tables — including this product — is consuming FBI-converted NIBRS
for the large majority of agencies. The conversion is not a detail: counting rules differ by
offense (crimes against persons are counted once per victim, motor vehicle theft once per
stolen vehicle, and the 2013 revised rape definition sums three incident codes rather than
one). The
repository implements and verifies each of these rules against the FBI's own converted totals;
on 2024 agencies that report both ways, the reconstruction matches the FBI's converted counts
in 98.8 percent of agency-offense cells.

Source selection follows a fixed priority: the FBI's frozen annual publication first, then a
local agency's own published annual figure, then a state publication, then the summary master
file, then the repository's own NIBRS-to-summary rollup for agencies the other sources miss.
The publication is preferred over the living master file as the frozen, auditable public
record.

A reported zero and a missing report are different facts. The incident-based rollup lane
emits an explicit zero for every Part I offense absent from an agency-year it actually covers,
over the months that lane reported — the semantics the summary lane always carried — so an
agency that recorded no robberies is recorded as having recorded none, rather than as having
said nothing. One source lane is then selected per agency-year and the count and the
months-reported are taken from that same lane, so a count from one source is never divided by a
coverage window from another. And an agency that neither reported in the reference year nor
carries usable history of its own does not receive a fabricated agency total: no peer-median
fill, no rate-median fill, and no invented zero. Its territory falls through to the surrounding
remainder and, where eligible, the benchmark process below.

Florida and New York use statewide publication contracts rather than a per-agency patchwork.
Florida's totals come from the Florida Department of Law Enforcement; New York's come from the
Division of Criminal Justice Services, including its full-year-scale figures for some agencies
that report only partially to the FBI. Mississippi TOPS contributes a smaller
state-publication lane under the same fixed source priority.

### Where no agency reports: the FBI benchmark

A bottom-up build creates no crime for territory where no agency reported, making that territory
appear nearly crime-free. The current release benchmarks that territory against an external
estimate.

The rule is one accounting identity per state and offense. Take the FBI's own published state
estimate — the Crime Data Explorer figure, which estimates the whole state including agencies
that did not report — and subtract everything already locked in from agencies that did. What
remains is the most missing crime the benchmark will concede. That residual is divided among the
units that are actually silent in proportion to a partially pooled rate model fitted across
state, agency type, and urbanicity. Each unit receives the smaller of what the model expects and
what the benchmark allows; where the model asks for less, the unused headroom is reported rather
than absorbed. In v21 this adds 96,840 expected offenses across 1,130 silent units covering
8.80 million residents. No agency-level observation is created anywhere in this: it is a device
for splitting a state target across territory. Every affected block group publishes the fraction
of its expected count that came from it, offense by offense, and above one half that cell's
confidence tier is forced to low.

Where agencies are silent, the surface is benchmarked to the FBI's own estimation program and
inherits its errors.

In eastern Kentucky, agency filings collapsed and the FBI state estimate absorbed the same
collapse. Kentucky is excluded from benchmark imputation in this release. The map there partly
reflects the geography of reporting rather than crime alone.

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
St. Louis, Tucson, and Washington, DC. For every other block group, there is no local incident
feed.

The second source is a national covariate model: a gradient-boosted regression (a
tree-ensemble method) trained to predict where offenses concentrate within a jurisdiction from
a classified pool of 346 block-group characteristics. These features describe commercial and
retail activity, land cover and land use, road and transit structure, population density,
housing, and socioeconomic composition drawn from the American Community Survey, employment
data from the Census Bureau's LODES program, points of interest from Overture, land cover from
the National Land Cover Database, and road and transit geography. The within-jurisdiction
allocation model draws 228 candidate features from the permitted classes of that pool after a
policy screen described below, applied uniformly to all seven offenses.

Where a city has usable incident data, the two sources are combined in a posterior: the
observed incident distribution is treated as evidence and blended with the model's prediction,
with the weight on the incident data determined by how well the feed reconciles to the
jurisdiction's official total. A feed whose annual counts closely match the official total is
trusted heavily and dominates the block-group shares; a feed that under-reports, over-reports,
or is sparse is pulled back toward the model. Where a city has no incident data — most of the
country — the shares come from the model alone.

How far the covariate model is used in uncovered areas depends on held-out evidence by offense.
Robbery, aggravated assault, larceny, and motor vehicle theft use the full covariate signal,
because the model demonstrably beats a
simple population or exposure baseline out of sample for these offenses. Burglary uses a
calibrated partial transfer, selected by a one-standard-error parsimony rule on cross-validated
error. Murder and rape use only a population or exposure baseline in uncovered areas because
their within-jurisdiction covariate signal does not generalize out of sample.

A bounded, distance-decaying adjustment carries a covered city's signal a short way into its
immediate surroundings and is then re-raked, so that a coverage boundary does not appear as an
abrupt seam without letting a city rewrite its rural neighbors.

### Denominators, reliability, and suppression

Every primary rate is an expected count divided by an offense-specific exposure measure.
Murder, rape, robbery, aggravated assault, and larceny use person exposure: the larger of a
jobs-based daytime proxy and LandScan daytime population, with a bounded cap where headquarters
employment would otherwise dominate. Burglary uses premises exposure: households plus calibrated
destination points of interest, retail jobs, and manufacturing, wholesale, transportation, and
warehousing jobs. Motor vehicle theft uses vehicle exposure: household vehicles plus workplace
jobs multiplied by the county's auto-commute share. A resident-normalized secondary index is
also published. The denominator system therefore distinguishes resident population,
daytime-and-jobs person exposure, premises, vehicles, and destination activity by offense;
destination points are a calibrated component of premises exposure rather than a land-area
proxy. Land area appears only in the density layers; it is not a risk denominator.

The output carries the disclosure with the point estimate. `estimate_mode` records whether the
index is count-derived or suppressed; `numerator_support_source` distinguishes direct city
incident evidence from model-only allocation; `reliability_tier` and
`recommended_display_geography` state the supported reading scale; and the exposure and
transient-use fields identify low-exposure cases. Large, low-population polygons can therefore
carry extreme point estimates while also carrying low reliability and a tract-or-larger
recommendation.

Suppression is offense-specific. A geography with fewer than 10 households is
`non_residential`; Census 98-series special-use tracts are `special_use`; person- and
vehicle-exposure rates below 50 units are `insufficient_exposure`; and burglary also suppresses
where premises exposure is below 10. A custom footprint whose reported mass cannot be paired
with adequate ambient exposure uses the same `insufficient_exposure` display status. Suppression
nulls the affected rate and index, not the expected count: counts remain available where needed
to conserve tract, jurisdiction, state, and national totals.

### The no-redlining screen

Because the neighborhood model uses socioeconomic features, there is a real risk of encoding
demographic composition as if it were crime risk. The build addresses this with an explicit
feature policy. Direct demographic composition features — race, ethnicity, ancestry, language,
nativity, sex composition, family structure, and national-origin proxies — are excluded from
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
allocation model. Excluding direct demographic fields does not remove all correlation with them.

## Handling defective inputs

Reporting artifacts in public crime data can create false neighborhood hotspots or artificially
quiet areas. The build detects these artifacts and falls back to the model or a coarser total.

Reporting-gap fills. Some agencies, particularly county sheriffs, report to the FBI for only
part of a year or not at all, which would make their jurisdictions appear nearly crime-free. A
per-agency fill estimates the missing counts from that agency's own reporting history — and
only its own, and only within the release's recency bound — rather than letting the gap
propagate. A source year is treated as partial only when the selected source lane's own coverage
record says it is partial.

Masked-gap detection. More insidious than an obvious gap is an agency whose completeness flags
read as clean while its counts are materially incomplete — for instance, a year that reports
all twelve months but at half the agency's normal volume. A detector reclassifies these cases
into the same fill process. Adjudicated zero-versus-missing, token-reporter, duplicate-identity,
and agency-succession cases are consumed from fail-closed registries before the totals are built.

Defunct and never-reporting jurisdictions. A municipal police department can still appear as a
jurisdiction on the map with no real reporting behind it at all — the department was dissolved,
absorbed into a county or neighboring city's force, or simply never filed a report, so every
offense shows zero counts and zero reported months in every year of the observation window.
Taken at face value that reads as an unusually safe place; it is actually a roster artifact, an
entry for a department that no longer functions as a source of totals, not evidence about the
crime rate. The fix is a single normalization rule: a municipal jurisdiction with zero reported
months across its entire observation window is not treated as a valid totals source, and its
block groups fall through to the surrounding county or state coverage exactly as unincorporated
territory would. The current release splits that predicate against the FBI's own 2024 agency
roster: an identifier the FBI still lists is a non-reporter rather than a dissolved department,
and its territory is eligible for the benchmark-constrained imputation above instead of simply
dissolving into the county.

State-publication lanes. Florida and New York use statewide publication contracts, and
Mississippi contributes TOPS rows under the fixed source priority described above.

Agency footprints. The v21 crosswalk resolves consolidated city-county agencies, state-police
districts, reviewed custom footprints, tribal reservation displacement, and registered
concurrent-jurisdiction carve-outs before allocation. A per-block-group ladder chooses an
activity, exposure, population, or area basis according to the evidence available for that
footprint. Custom-footprint rules fail closed when their configured footprint rows are absent.
The result is still an approximation of service geography, not a guarantee that every event
occurred in the receiving block group.

Coordinate quarantine and reviewed exceptions. Incident feeds frequently place records at
default or masked coordinates: a police headquarters, a precinct centroid, a hospital where a
report was taken. Left alone, these create enormous false hotspots. A quarantine registry
removes named artifact coordinates from the incident evidence before aggregation (currently 108
entries), and a separate tripwire fails the release if any single point carries an implausible
share of a city's located incidents unless that point has been individually reviewed and
whitelisted as a genuine concentration — a mall, a transit hub, a large apartment complex (119
reviewed exceptions).

Offense-level exclusions. When a feed's count for one offense cannot be reconciled and the
discrepancy is not explained by documented source omissions, that offense is excluded for that
city and its allocation falls back to the model — a fail-closed default. Admission is made
offense by offense, so rejecting one feed/offense does not silently reject or admit the rest.

Rape texture defaults to deny. Rape incident locations are pervasively masked in public feeds —
coded to precinct station houses or to hospitals rather than to where the offense occurred. In
New York, 9,932 located rapes fell on exactly 77 precinct points. Because these
locations are systematically misleading, the product does not use direct rape point locations
for neighborhood texture by default; it allows them only for four cities whose rape geography
was individually audited and found diffuse and unmasked. Everywhere else, rape allocation falls
back to the baseline.

## How the surface is validated

The fail-closed release gate runs automated checks against the actual rendered output
files and map tiles rather than an intermediate; a build that fails any of them cannot be
promoted, and the current release passes every one with zero open issues. Two further checks —
the held-out cross-validation comparison described below and a visual inspection of the rendered
surface — are computed and recorded for every release as part of the promotion process, but they
are reviewed by a person and are not among the automated checks that mechanically block
promotion.

The gate has several blocks. Total-lane integrity confirms that jurisdiction controls reconcile
and that no incident feed has been promoted to a total without reconciling to the official
figure. Allocation coherence confirms that block-group counts sum to their controls, that tract
counts equal the sum of their block groups, that no count is negative, and that exactly one
share source is used per cell. Index coherence recomputes every published rate and index from
its expected count and the stored national normalizer and fails if any value was written
independently — the check that makes count-and-index incoherence impossible. Spatial-artifact
checks look for the failure modes that a crime map is prone to: state-border discontinuities,
tract-level flatness, checkerboard speckle, low-denominator hotspots, and coverage seams. A
feature-policy audit confirms the no-redlining exclusions hold in the fitted model.

Held-out cross-validation measures within-city allocation. The
35 cities with incident data are used as a test set: the model is repeatedly retrained with one
city withheld, and its predicted neighborhood distribution for that city is scored against the
city's actual incident distribution. The scoring metric is total variation distance (TVD)
between the predicted and observed share distributions across the city's block groups. TVD
ranges from 0 (identical distributions) to 1 (no overlap); a TVD of 0.30 means that about 30
percent of the predicted incident mass would have to be moved between block groups to match
the observed distribution, and about 70 percent already overlaps. Because this is measured on
withheld cities, it estimates how the model performs where it has no local data — which is the
situation for most of the country.

The held-out, incident-weighted TVD for the current release, by offense:

| Offense | Held-out TVD | Cities scored |
|---|---|---|
| Motor vehicle theft | 0.29 | 33 |
| Larceny-theft | 0.31 | 34 |
| Aggravated assault | 0.33 | 28 |
| Burglary | 0.33 | 33 |
| Robbery | 0.40 | 33 |
| Rape | 0.63 | 4 |
| Murder | 0.73 | 27 |

The standard errors on these figures range from about 0.008 to 0.026. The release standard is
non-degradation: no offense may
worsen against a frozen baseline by more than one standard error, and none does. Measured
like-for-like against the frozen evidence set, the current release's scores have zero change
for every offense: the last two releases changed the totals lane only, leaving the share
model's inputs unchanged bit for bit. This cross-validation measures within-city allocation and
is blind by construction to everything the totals lane does — the zero semantics, the
silent-agency rule,
and the benchmark-constrained imputation above are all invisible to it, and were gated on the
release validator, the release-over-release outlier screen, and human visual review instead.
The rape figure rests on only four cities — the four with audited, unmasked rape geography —
and should be read as low-confidence.

The visual inspection stage renders the new surface alongside a set of unchanged control cities
and confirms that the intended geography appears and that untouched areas do not move; like the
cross-validation comparison, it is performed and recorded for each release but is a human
judgment rather than an automated block. After the surface is promoted, the full automated
validator is run once more against the live output directory.

The surface also undergoes recurring cold review. Claims are verified against source records at
block-group level; material findings are fixed or recorded as limitations.

## Pre-registration

Before any new city's incident data is examined, the model's predicted neighborhood
distribution for that city — the prediction it makes with no knowledge of the city's actual
incidents — is frozen and committed,
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
frozen bands — five better, two at the band edge — the strongest prospective result of any
onboarded city. Pre-registration converts each new city from a tuning opportunity into an
out-of-sample test of the error bars quoted above.

## What the accuracy numbers mean for a user

For the high-volume offenses — larceny, motor vehicle theft, aggravated assault, burglary — the
model reproduces roughly two-thirds to
seventy percent of a city's neighborhood distribution in cities it has never seen. In practice
that means it reliably captures the broad within-city gradient: which parts of a city carry
more property crime and which carry less, and the general shape of the concentration. It does
not mean block-group precision. A single block group's index can be off substantially even
where the citywide pattern is right, and users should treat an individual block group's value
as an estimate with real uncertainty, not a measurement.

For robbery the model is somewhat weaker (TVD around 0.40), and for murder and rape it is weak
at the block-group level. Murder and rape are rare events; in most neighborhoods the expected
count is small, and small counts are volatile.

For these two offenses the product coarsens the published support rather than the estimator.
Their per-offense index and rate are published at the census tract and coarser; at the block
group they are withheld, because a single year of murder or rape there is Poisson noise on a
model prior, not a stable rate. The block group keeps the expected count for reconciliation from
block group to tract to nation, but no per-offense index or rate for these two offenses.
Where incidents are observed, a zero-or-one annual block-group count is a fact, but a rate built
on it would be noise. Neighborhood-level violence rankings are temporally stable at coarser scales, while
single-year block-group estimates of rare events require shrinkage or coarser support. The
published support is the census tract.

The aggregate indexes still resolve at the block group, but they take their murder and rape terms
at tract support: the harm-weighted index draws the two offenses as the tract count spread across
the tract by each block group's share of person exposure, and the offense-averaged indexes draw
them from the parent tract's per-offense value. The severity weights are untouched — an aggregate
that is meant to be sensitive to the most serious offenses stays so — and every tract, jurisdiction,
and national total still reconciles, because within-tract redistribution conserves the tract count
exactly. For every cell the reliability metadata remains: the direct incident support behind the
value, Poisson-based reliability intervals, and a reliability tier. A forward-looking
projected-risk product would be a separate, explicitly labeled surface built on the internal model
prior and validated temporally; it is future work, and it — not the descriptive indexes here — is
where finer-grained display of rare offenses would belong.

## How the map is shaded

The published map uses one fixed, value-anchored break set — 12.5 to 800, log-symmetric about
100, with white at exactly 100, the national reference —
identical across every index layer and interpolated in log space, so a given shade means the same
multiple of the national rate on burglary as on robbery and the layers can be compared to each
other. Both ends are open. The legend reports, at each break, the share of people and the share
of land above it. Large rural polygons can dominate a
screenshot by area while representing little population.

The density layers paint land rather than population, so their stops are drawn from
area-weighted percentiles and the density field is carried at four decimal places. The CARTO
label overlay is separate basemap imagery: labels and their baked halos are not data geometry.
At national zoom the halo/downsampling residual can darken warm colors in dense labeled cores;
the underlying polygon value, popup, and legend are unchanged.

## Known limitations

Rare-offense noise. Murder and rape carry substantial block-group noise, which is why their
per-offense index and rate publish at census-tract support rather than block-group support. The
block-group expected counts remain for reconciliation, and block-group aggregates use
tract-supported murder and rape contributions.

Modeled texture. Outside the 35 direct-incident cities, within-jurisdiction texture is modeled
and raked to the control total. Small-town and rural neighborhoods are extrapolation beyond the
city validation support. The 3.77 percent model-only outlier class is published without
automatic muting; `numerator_support_source`, reliability tier, and recommended geography
identify these cells. They can produce visually extreme values, especially on large rural
polygons.

Source limitations. The surface inherits under-reporting and definitional differences in its
official sources. Eastern Kentucky remains the documented benchmark-inherited reporting
collapse. Token-reporter, Class-J, and `no_agency_evidence` territory remains in Indiana,
Nebraska, Connecticut, Savannah, and Osceola; those values can be much too low even though they
follow the recorded source. Chicago aggravated assault remains a separate definitional split
between aggravated assault and aggravated battery; v21 does not replace the federal control.

Footprint limitations. Some policing responsibility remains unresolved in
concurrent-jurisdiction and PL-280 counties. The Lakeside-style micro-municipality class and
near-floor component queue can attribute an agency total to a footprint whose represented
population is too small or incomplete. Remaining tribal concurrency cases can split one
reservation across tribal, county, state, and state-control systems. These are placement limits:
conservation of the control total does not prove the event's block-group location.

Exposure limitations. No public denominator used here fully measures tourists, visitors, transit
riders, or other transient population. Visitor-heavy and very low-population polygons can
therefore overstate per-person risk even after the daytime, premises, vehicle, destination, and
suppression rules are applied. Large polygons also dominate screenshots by land area rather than
by population or evidentiary support.

Temporal reference. The jurisdiction totals are 2024. Direct incident shares pool multiple
years for spatial stability and are then raked to 2024 levels, so a neighborhood that changed
sharply near the end of the window can carry stale texture. No forward projection ships in this
surface.

State and county boundaries. Adjoining areas are not constrained to be continuous. They can use
different source contracts, state definitions, jurisdiction totals, footprints, and modeled
shares. Continuity across those boundaries is not imposed.

Interpretation. CrimeRisk estimates association in police-recorded data, not causation. It does
not identify why a place has a given value, guarantee the within-jurisdiction location of an
event, measure unreported crime, or provide a visitor-complete denominator. Alaska, Hawaii, the
territories, and special federal jurisdictions remain out of scope. The surface is an open
public-data estimate, not a one-for-one commercial replica.

## Reproducibility

The same inputs and configuration reproduce the same outputs.

Pinned sources with recorded provenance. The target-year FBI CIUS, NIBRS, Return A, and Crime
Data Explorer inputs are 2024; the Kaplan annual and incident archives run through 2024.
Florida FDLE, Mississippi TOPS, and New York DCJS target rows are also 2024, with the DCJS
extract last refreshed on 2025-11-25. Local incident windows vary by city and the calibration
evidence spans 2018–2024. Geometry is 2020 TIGER/Line. Exposure and covariate vintages include
ACS 2020–2024 five-year, Census Vintage 2025 population estimates using its 2024 estimate,
LODES8, LandScan USA 2021 daytime population, NLCD 2023, 2024 HPMS, and Overture Places release
2026-06-17.0. The exact file identities and city-specific windows are recorded in the input and
build manifests. The federal conversion rules the totals depend on are verified against the
FBI's own converted counts and documented rule by rule.

Refresh rule. CrimeRisk advances annually to the most recent complete reference year only after
the totals, share-model, validator, visual, and first-look gates pass. A refresh changes the
point-in-time surface; it does not turn prior values into forecasts.

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

- Surface size, national totals, and per-offense published counts: the release validation
  summary and the block-group and tract output parquet files
  (`crimerisk_block_group_2024_ags_core.parquet`, 238,193 rows;
  `crimerisk_tract_2024_ags_core.parquet`, 83,776 rows; national expected-count total
  7,178,828.700; FBI-calibrated companion total 7,158,419).
- Zero-versus-missing semantics, lane coherence, and the silent-agency rule: the agency
  zero-semantics tests (`tests/test_agency_zero_semantics.py`) and the fill-mass regression
  ratchet (`configs/agency_fill_mass_baseline.json`).
- Benchmark-constrained imputation: `src/crimerisk/benchmark_imputation.py` and
  `tests/test_benchmark_imputation.py`; the per-state identity, unit counts, conflict-kind
  tallies, and the recorded reconciliation caveats in
  `state/controls/benchmark_imputation_2024.json`; per-unit sub-targets in
  `benchmark_imputation_units_2024.parquet`; the state-by-offense ledger, including
  `benchmark_unused_headroom` and `benchmark_conflict_kind`, in
  `state/controls/state_control_comparison.parquet`. The published per-cell shares are the
  `benchmark_imputed_share_*` fields on the block-group and tract surfaces, with
  `benchmarked_nonreporter_imputation` appearing in `confidence_reasons_*`.
- The published color scale and density stops: `INDEX_BREAKS` in `frontend/public/index.html`
  and the stop construction, with its recorded rationale, in
  `frontend/build/01_extract_indices.py`.
- Field and index definitions, denominators, and aggregate rules: the published-field policy
  and the output build manifest, which records every normalizer.
- Federal data mechanics, conversion rules, and the 98.8 percent reconciliation:
  `docs/FBI-DATA-GUIDE.md` and `scripts/diagnostics/verify_fbi_conversion_rules.py`.
- The 35 direct-evidence cities and their held-out scores: the nested cross-validation evidence
  in the release candidate directory (`nested_city_cv_v21.parquet` with its companion JSON, and
  `nested_cv_v21_vs_v17.csv`, which records the zero delta against the frozen evidence set).
- Covariate feature policy and the no-redlining exclusions: the feature-transfer policy artifact
  (`feature_transfer_policy_2024.parquet`: 346 classified features — 158 both-axes, 91
  between-only, 69 proxy-review, 27 excluded-protected, 1 unstable) and the residual
  feature-policy block of the build manifest (228 candidate features, uniform across all seven
  offenses).
- Data-hygiene mechanisms: the reporting-gap and masked-gap fills, the defunct-jurisdiction
  normalization rule (`source_selection.py`), the state-publication lanes and their non-report
  rule (`observations.py`), the adjudicated agency-to-jurisdiction overrides
  (`configs/local_resolution_overrides.csv`, each row carrying its evidence), the
  consolidated-agency
  footprints (`configs/consolidated_agency_footprints.csv`), the coordinate quarantine
  (`configs/city_feed_coordinate_quarantine.csv`, 108 entries) and exact-point exceptions
  (`configs/city_feed_exact_point_exceptions.csv`, 119 entries), the offense-level admissions
  (`configs/gate17_city_offense_admissions.csv`), and the rape texture policy
  (`configs/city_offense_texture_policy.csv`).
- The release gate: `scripts/diagnostics/validate_release_outputs.py` and the promotion script's
  hash and ancestry checks.
- The v21 promotion and display records: `state/candidates/stage-program-v21/` and
  `state/qa/stage6_screen/`; source vintages are in
  `state/reference/input_manifest.json`.
- Pre-registration: the frozen prediction artifacts committed before each onboarding round and
  the per-city scoring records in each city's review packet.
