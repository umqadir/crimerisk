# CrimeRisk

CrimeRisk is a national, block-group-level crime-risk surface for 2024, covering the
forty-eight contiguous states and the District of Columbia. It reports indices (100 =
national average), expected counts, and density layers for the FBI's seven Part I offenses.
The surface is built from official FBI and state jurisdiction crime totals, which are then
distributed within each jurisdiction by a covariate model blended with audited city incident
data where it exists. It is an open, public-data reconstruction in the spirit of commercial
neighborhood crime-risk products such as Applied Geographic Solutions' CrimeRisk: a research
product, useful but weaker than a commercial offering, and not a tool for individual safety
decisions.

**Live map:** [umqadir.github.io/crimerisk-map](https://umqadir.github.io/crimerisk-map/)

## Methodology

The full methodology, validation results, and known limitations are documented in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md). That document is the authoritative reference for
how the surface is built and how well it performs; read it before relying on the data.
[docs/PIPELINE.md](docs/PIPELINE.md) describes the build as a pipeline — each stage's inputs,
outputs, invariants, and manual-override registries, in application order. Upstream data
sources and their attribution and license terms are documented in
[docs/DATA_SOURCES_ATTRIBUTION.md](docs/DATA_SOURCES_ATTRIBUTION.md), and the mechanics of
the federal crime-data systems the build ingests are in
[docs/FBI-DATA-GUIDE.md](docs/FBI-DATA-GUIDE.md).

Two design points carry most of the weight. Jurisdiction totals and neighborhood shares are
separate lanes: official agency totals set each jurisdiction's level, and the model only
decides how that level distributes within the jurisdiction, so modeled texture can never
change how much crime a jurisdiction has. And the within-city model is scored by held-out
cross-validation against 35 cities with audited incident data — each city scored by a model
that never saw it — with new cities pre-registered: the prediction is frozen and committed
before the city's real data is ever examined.

## Viewing the map

The public map is at [umqadir.github.io/crimerisk-map](https://umqadir.github.io/crimerisk-map/).
The frontend itself is static, under `frontend/public/`, backed by PMTiles archives and a
snapshot manifest. The PMTiles archives are large binary outputs of the build and are not
included in this repository; they are produced by the `frontend/build/` scripts from a
released block-group output file. Once `frontend/public/` contains the PMTiles archives and
`snapshot.json`, serve it locally with:

```bash
uv run python frontend/serve.py 8777
```

Then open `http://127.0.0.1:8777/` in a browser. The server adds HTTP Range support, which
the map's PMTiles loader requires to read tiles directly out of the archive.

## Limitations

The surface estimates the geography of *police-recorded* crime: crime that is never reported
to police, or never recorded by them, is in neither the training data nor the validation
data. Murder and rape are rare events, so their per-offense index and rate are published at
census-tract support rather than the block group, where a single year of a rare offense is
noise on a model prior. For roughly nine in ten block groups there is no local incident feed,
so the within-jurisdiction distribution comes from the covariate model alone, validated only
against the cities that do have incident data. Where no agency reported at all, territory is
benchmarked against the FBI's own state estimates and marked as imputed — on the map itself,
not only in the metadata. The surface has no measure of transient or visitor population, so
per-resident rates in places dominated by daytime workers or tourists can overstate risk to
residents. See docs/METHODOLOGY.md for the full discussion, including the regions where the
estimate is known to be weak.

## Reproducing the pipeline

The build pipeline lives in `src/crimerisk`, driven through `main.py`:

```bash
uv sync
uv run python main.py build-release --emit-fbi-calibrated
```

This rebuilds the released outputs from the expected external inputs (FBI/NIBRS master
files, ACS, Census, LODES, Overture, NLCD, TIGER geometry, and city incident feeds). These raw
inputs are large and are not included in this repository; expect a substantial download and
multi-hour build.

The hand-reviewed decisions the build consumes — agency identity and zero-versus-missing
adjudications, footprint overrides, source contracts — are included as the registries under
`configs/`, applied fail-closed at named pipeline entry points (see docs/PIPELINE.md). The
research packets behind those decisions are part of the internal development workspace and are
not included, so a from-scratch run will not reproduce every input byte-for-byte; the code and
registries do reproduce the method in full, and the curated evidence backing the validation
numbers in docs/METHODOLOGY.md is included under `state/output/` and `state/modeling/`.

## Evidence and validation artifacts

`state/output/` and `state/modeling/` contain a curated subset of the pipeline's own build and
validation outputs: the release validation summary, the held-out nested cross-validation
results behind the accuracy table in docs/METHODOLOGY.md (including the current release's
comparison against the frozen evidence set), the pre-registration records for the two rounds
of city onboarding described there, and the feature-policy classification behind the
no-redlining screen. These are outputs, not manual inputs, and are regenerated by the build.

## License

Code is licensed under the MIT License; see [LICENSE](LICENSE).

Output data (the crime-risk indices, expected counts, and density layers) is derived from
public government sources (FBI UCR/NIBRS, state crime publications, and the U.S. Census
Bureau) and from other public and licensed datasets described in
[docs/DATA_SOURCES_ATTRIBUTION.md](docs/DATA_SOURCES_ATTRIBUTION.md). Retain the attributions
in that document when reusing or redistributing the released data.
