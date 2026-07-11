# Data Sources And Attribution

This product is a public-data modeled crime-risk surface for 2024 Census block groups
and tracts. Suggested citation:

> CrimeRisk public-data crime-risk estimates, 2024 release. Derived from FBI UCR/CDE
> crime data, U.S. Census Bureau demographic/geographic data, federal transportation,
> land-cover, education, health, and price-index datasets, Overture/OSM-derived place
> features, and selected municipal open-data incident feeds. Cite this product with
> this file and retain the upstream attributions below.

The release publishes modeled block-group and tract aggregates, not raw incident records
or raw source extracts. This file is an attribution inventory based on the repo source
contract (`src/crimerisk/required_inputs.py`), source registry
(`configs/city_incident_sources.csv`), and build code references. License notes use
standard known provider terms and are not legal advice.

## Legal-Review Flags Before Public Release

- **OpenStreetMap / ODbL-derived content:** OSM is ODbL. If any released database is
  legally a derivative database of OSM/ODbL content, attribution and share-alike
  obligations may apply. The repo scan found no direct raw OSM loader in the active
  required-input contract, but OSM is referenced as an Overture/POI upstream.
- **Overture Places:** Overture datasets are theme-specific and include CDLA-Permissive
  2.0 and/or ODbL-attributed components. Keep Overture attribution with published
  outputs and confirm whether the exact Places release used has any ODbL passthrough.
- **Municipal incident feeds and annual reports:** City open-data portals are generally
  public/open with attribution encouraged, but terms vary by Socrata, ArcGIS, CKAN,
  CARTO/OpenDataPhilly, Tableau dashboards, and city-hosted reports. Review before
  redistributing raw records. Derived aggregate publication is the intended use.
- **Florida FDLE FIBRS and Mississippi TOPS:** State publication feeds are public
  official crime-reporting surfaces, but portal-specific reuse terms should be reviewed
  before redistributing raw extracts.

## Source Inventory

| Source | Provider / URL | Provides | Used in build | License / attribution | Derived-aggregate constraint |
|---|---|---|---|---|---|
| FBI CDE, CIUS/RCN, UCR, SRS Return A, NIBRS, estimated crimes | FBI Crime Data Explorer, https://cde.ucr.cjis.gov; legacy CIUS pages at https://ucr.fbi.gov | Official annual crime controls, published local agency rows, NIBRS agency tables, estimated state totals | `observations`, `source_selection`, `controls`, FBI-calibrated output; see `docs/FBI-DATA-GUIDE.md` | U.S. Government public-domain data; cite FBI UCR/CDE and publication vintage | No known restriction on modeled aggregate redistribution; do not imply FBI endorsement |
| Kaplan openICPSR Return A and NIBRS concatenations | Jacob Kaplan openICPSR Return A project https://www.openicpsr.org/openicpsr/project/100707; NIBRS project https://www.openicpsr.org/openicpsr/project/118281 | Restacked FBI Return A and NIBRS master-file segments | `data/SRS-Kaplan-1960-2024/*`, `data/NIBRS-Kaplan-1991-2024/*`; SRS/NIBRS panel and rollups | Underlying data are FBI public-domain records; cite Kaplan/openICPSR packaging and vintage used | Aggregates can be published; retain source/citation chain |
| U.S. Census ACS 5-year | U.S. Census API / data.census.gov, e.g. https://api.census.gov/data/2024/acs/acs5 | Block-group and tract demographic, housing, vehicle, commuting, and socioeconomic covariates | `data/ACS-5yr-2020-2024/parsed/*`; BG prior and denominators | U.S. Government public domain; cite U.S. Census Bureau ACS 2020-2024 5-year | No restriction on derived aggregate redistribution |
| U.S. Census Population Estimates | U.S. Census Bureau Vintage 2025 county population totals, https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/counties/totals/co-est2025-alldata.csv | County/place/state population updates | Controls, feature build, output denominators | U.S. Government public domain; cite Census Population Estimates Program | No restriction on derived aggregate redistribution |
| U.S. Census TIGER/Line geometry and roads | U.S. Census TIGER/Line 2020 blocks, block groups, tracts, places, county subdivisions, roads, https://www2.census.gov/geo/tiger/TIGER2020/ | Jurisdiction geometry, small-area crosswalks, road metrics | `geometry`, `reference_layers`, `roads` covariates | U.S. Government public domain; cite U.S. Census Bureau TIGER/Line | No restriction on derived aggregate redistribution |
| LEHD LODES | U.S. Census LEHD LODES8, https://lehd.ces.census.gov/data/lodes/LODES8 | Workplace-area jobs, resident/workplace flows, activity-denominator support | `data/LODES/parsed/lodes_wac_block_groups.parquet`; BG prior and exposure proxy | U.S. Government public domain with Census/LEHD attribution customary | No restriction on derived aggregate redistribution |
| LandScan USA 2021 | Oak Ridge National Laboratory, https://landscan.ornl.gov | Modeled daytime population aggregated to 2020 block groups | `data/LandScan-USA/block_group_landscan_usa_2021.parquet`; person-exposure denominator floor-lifter for murder, rape, robbery, aggravated assault, and larceny | LandScan USA 2021, ORNL; Creative Commons Attribution 4.0 International (CC BY 4.0); cite ORNL/LandScan USA vintage | Derived aggregates can be published with clear ORNL/LandScan attribution; LandScan USA excludes transitory populations such as tourists |
| Overture Places | Overture Maps Foundation STAC/S3, https://stac.overturemaps.org/catalog.json | Consumer destinations and commercial-core place features | `data/Overture-Places/parsed/*`; residual allocator features and burglary commercial-premises component | Overture per-theme terms, typically CDLA-Permissive 2.0 with attribution; some components may carry ODbL obligations | Publish derived aggregates with Overture attribution; review ODbL passthrough/share-alike before release |
| OpenStreetMap | OpenStreetMap contributors, https://www.openstreetmap.org/copyright | OSM-derived place/map facts, primarily via Overture attribution lineage in this repo scan | Referenced as OSM/POI upstream; no active required raw OSM input found | ODbL; requires attribution and may require share-alike for derivative databases | Legal review if any released database is derived from OSM/ODbL data |
| NLCD / MRLC | Multi-Resolution Land Characteristics consortium / USGS, https://www.mrlc.gov | 2023 land cover, imperviousness, impervious descriptor rasters | `data/NLCD/parsed/block_group_nlcd_2023.parquet`; BG prior covariates | U.S. Government public domain; cite USGS/MRLC NLCD | No restriction on derived aggregate redistribution |
| FHWA HPMS | Federal Highway Administration Highway Performance Monitoring System, https://www.fhwa.dot.gov/policyinformation/hpms.cfm and https://geo.dot.gov | Highway and traffic-exposure metrics | `data/HPMS/parsed/block_group_hpms_2024.parquet`; BG prior covariates | U.S. Government public domain; cite FHWA HPMS | No restriction on derived aggregate redistribution |
| National Transit Map (NTM) / FTA/BTS | National Transportation Geospatial Platform ArcGIS item, https://ngda-transportation-geoplatform.hub.arcgis.com/api/download/v1/items/959b4adc3ff94bc6a46f2c1d515d09aa/shapefile?layers=0 | Transit stop/access features | `data/NTM/parsed/block_group_transit_stops.parquet`; transit covariates | U.S. Government public domain; cite USDOT/FTA/BTS National Transit Map | No restriction on derived aggregate redistribution |
| CPI-U | BLS CPI-U series accessed via FRED CSV, https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL | Inflation adjustment for ACS monetary fields | `state/cache/cpi/CPIAUCSL.csv`; covariate feature build | BLS federal data are public domain; FRED attribution customary for access copy | No restriction on derived aggregate redistribution |
| NCES EDGE | National Center for Education Statistics EDGE, https://nces.ed.gov/programs/edge/data | Public school and postsecondary location anchors | `data/NCES-EDGE/parsed/block_group_education_anchors_2425.parquet`; BG prior covariates | U.S. Government public domain; cite NCES EDGE | No restriction on derived aggregate redistribution |
| CMS Hospital General Information | Centers for Medicare & Medicaid Services provider data, https://data.cms.gov/provider-data/ | Hospital anchor locations and nearest-hospital features | `data/CMS-Hospital-General-Info/parsed/block_group_hospital_anchors.parquet`; BG prior covariates | U.S. Government public domain; cite CMS Provider Data | No restriction on derived aggregate redistribution |
| Florida FDLE FIBRS/UCR | Florida Department of Law Enforcement, https://www.fdle.state.fl.us/CJAB/UCR | 2024 state-publication annual offense rows for Florida | `state_publication_annual` lane; required because Florida is absent from FBI national files for 2024 | Public official state data; attribution to FDLE required/customary; portal terms should be checked | Derived aggregates intended; review before raw extract redistribution |
| Mississippi TOPS | Mississippi Crime Statistics / Mississippi DPS TOPS, https://mscrimestats.dps.ms.gov/public/View/dispview.aspx | Optional 2024 state-publication report cache | Optional `state_publication_annual` support | Public official state data; attribution to Mississippi TOPS/state source required/customary; portal terms should be checked | Derived aggregates intended; review before raw extract redistribution |
| Municipal annual publication packets | City/state/local agency annual reports, tracked in the internal development workspace's review packets (not included in this repository) | Reviewed direct local annual publication rows | `local_publication_annual` lane when promoted | Public official reports; attribution to issuing agency/source required/customary; terms vary | Derived aggregate use intended; review any raw report extract redistribution |

## Municipal Incident Feeds In `configs/city_incident_sources.csv`

These feeds support city incident share surfaces, validation, or counts-only packet work.
They are not published as raw incident records in the release.

| City | Portal / source | URL(s) | Disposition | License / attribution note |
|---|---|---|---|---|
| New York, NY | NYPD Complaint Data Historic + Current feeds, Socrata dual feed | https://data.cityofnewyork.us/api/views/qgea-i56i/rows.csv?accessType=DOWNLOAD; https://data.cityofnewyork.us/api/views/5uac-w243/rows.csv?accessType=DOWNLOAD | ready_now | NYC open data; attribution to NYC Open Data/NYPD; portal terms vary |
| Chicago, IL | Chicago Crimes 2001 to Present, Socrata | https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD | ready_now | Chicago open data; attribution to City of Chicago/CPD; portal terms vary |
| Boston, MA | Analyze Boston / BPD Crime Incident Reports, CKAN CSV | https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system | ready_now | Boston open data; attribution to City of Boston/BPD; portal terms vary |
| Seattle, WA | Seattle SPD Crime Data, Socrata | https://data.seattle.gov/api/views/tazs-3rd5/rows.csv?accessType=DOWNLOAD | ready_now | Seattle open data; attribution to City of Seattle/SPD; portal terms vary |
| San Francisco, CA | SFPD incident early + late feeds, Socrata dual feed | https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD; https://data.sfgov.org/api/views/wg3w-h783/rows.csv?accessType=DOWNLOAD | ready_now | DataSF/SFPD open data; attribution required/customary; portal terms vary |
| Austin, TX | Austin APD Crime Reports, Socrata | https://data.austintexas.gov/api/views/fdj4-gpfu/rows.csv?accessType=DOWNLOAD | ready_now | Austin open data; attribution to City of Austin/APD; portal terms vary |
| Mesa, AZ | Mesa Police incidents 2020-present, Socrata CSV | https://data.mesaaz.gov/resource/hpbg-2wph.csv | ready_now | Mesa open data; attribution to City of Mesa/Police; portal terms vary |
| Baltimore, MD | Legacy SRS Part 1 + NIBRS Group A companion, ArcGIS dual feed | https://services1.arcgis.com/UWYHeuuJISiGmgXx/arcgis/rest/services/Part1_Crime_Beta/FeatureServer/0; https://services1.arcgis.com/UWYHeuuJISiGmgXx/arcgis/rest/services/NIBRS_GroupA_Crime_Data/FeatureServer/0 | ready_now | Baltimore/ArcGIS open data; attribution to city/BPD; portal terms vary |
| Philadelphia, PA | OpenDataPhilly crime incidents + CARTO SQL API | https://opendataphilly.org/datasets/crime-incidents/; https://phl.carto.com/api/v2/sql | ready_now | OpenDataPhilly/Philadelphia Police data; attribution required/customary; terms vary |
| Los Angeles, CA | LAPD legacy geocoded incidents + public NIBRS companion feeds, Socrata multiplex | https://data.lacity.org/api/views/63jg-8b9z/rows.csv?accessType=DOWNLOAD; https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD; https://data.lacity.org/api/views/y8y3-fqfu/rows.csv?accessType=DOWNLOAD; https://data.lacity.org/api/views/gqf2-vm2j/rows.csv?accessType=DOWNLOAD | blocked | LA open data; blocked for share promotion; review terms before any raw redistribution |
| Washington, DC | MPD Crime Incidents 2024 + official anchors, ArcGIS plus reports | https://opendata.dc.gov/datasets/DCGIS::crime-incidents-in-2024; https://mpdc.dc.gov/dailycrime; https://mpdc.dc.gov/page/mpd-annual-report-2024 | ready_now | DC open data/MPD reports; attribution to DC/MPD; terms vary |
| Denver, CO | Denver crime FeatureServer + dashboard/archive anchors, ArcGIS | https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/ODC_CRIME_OFFENSES_P/FeatureServer/324; https://www.denvergov.org/opendata/dataset/city-and-county-of-denver-crime | ready_now | Denver open data; attribution to City and County of Denver/DPD; terms vary |
| Minneapolis, MN | Crime_Data FeatureServer + official dashboard documents, ArcGIS | https://services.arcgis.com/afSMGVsC7QlRK1kZ/arcgis/rest/services/Crime_Data/FeatureServer/0; https://opendata.minneapolismn.gov/datasets/cityoflakes::crime-data/about | ready_now | Minneapolis open data; attribution to City of Minneapolis/MPD; terms vary |
| Detroit, MI | Crime Viewer, RMS Crime Incidents 2024, DPD year-end stats, ArcGIS/PDF | https://data.detroitmi.gov/pages/crime-viewer/; https://data.detroitmi.gov/datasets/detroitmi::rms-crime-incidents/about; https://services2.arcgis.com/qvkbeam7Wirps6zC/arcgis/rest/services/RMS_Crime_Incidents_2024/FeatureServer | counts_usable_not_city_share_promotable | Detroit open data/DPD report; counts-only in repo; review before raw redistribution |
| Nashville, TN | MNPD UCR dashboard + crime-statistics reports, Tableau/reports | https://www.nashville.gov/departments/police/data-dashboard/ucr-incidents-map; https://www.nashville.gov/departments/police/download-police-dashboard-data; https://policepublicdata.nashville.gov/t/Police/views/UCRPartICounts_16286288901850/IncidentsMap | counts_usable_not_city_share_promotable | Nashville/MNPD public dashboard; counts-only in repo; Tableau/report terms vary |
| Portland, OR | PPB reported crime open data + Tableau year downloads + annual report | https://www.portland.gov/police/open-data/reported-crime-data; https://public.tableau.com/views/MonthlyReportedCrimeStatistics/DownloadOpenData?:showVizHome=no | counts_usable_not_city_share_promotable | Portland/PPB public data; counts-only in repo; Tableau/report terms vary |

## Summary License Classes

- **U.S. federal datasets:** FBI, Census, LEHD/LODES, TIGER, NLCD/USGS/MRLC, FHWA HPMS,
  NTM, BLS CPI, NCES, and CMS are treated as U.S. Government public-domain data with
  source attribution customary and no known restriction on derived aggregates.
- **LandScan USA:** ORNL LandScan USA 2021 is CC BY 4.0. Retain ORNL/LandScan USA
  attribution and the 2021 vintage note when publishing derived outputs.
- **Overture / OSM-derived datasets:** retain attribution; review ODbL/share-alike
  implications before public database release.
- **State, municipal, and portal datasets:** public/open official data with attribution
  encouraged or required; raw-record redistribution terms vary by portal. This product's
  intended release surface is modeled aggregate output, not raw records.
