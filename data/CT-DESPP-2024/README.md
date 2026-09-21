# CT DESPP 2024 town source

## Provenance

| Field | Value |
|---|---|
| Publisher | Connecticut Department of Emergency Services and Public Protection |
| Report site | https://ct.beyond2020.com/CT_public |
| Report | Crimes by Offense and Town |
| View | 381 |
| Retrieval date | 2026-08-02 |
| Export method | Site download endpoint replay |
| Export format | CSV list format |
| Data source header | Connecticut_SS, Offense Data |
| Slicer | Jurisdiction by Type = Connecticut; all types |
| Members | All towns + all jurisdictions |
| Measure | Number of Crimes |
| Export years | 2021-2026 |
| Raw file | `raw/ct_crimes_by_offense_and_town_2021_2026_list.csv` |
| Parser | `scripts/pull/parse_ct_despp_town_srs.py` |
| Parsed file | `parsed/ct_despp_town_srs_2024.csv` |
| Parsed year | 2024 |
| Parsed rows | 539 |
| Parsed units | 77 |
| Production lane units | 70 |
| Annual report | https://portal.ct.gov/despp/-/media/despp-beta/pdf/data/crime-in-connecticut-2024-final.pdf |
| SRS supplement | https://portal.ct.gov/despp/-/media/despp-beta/pdf/data/srs-crime-in-ct-2015-2024final.pdf |

## Row selection

| Parameter | Value |
|---|---|
| Geography member | `CSP - <Town>` |
| Geography alias | `CSP - Barkhamstead` → Barkhamsted |
| Bare-town reconciliation alias | Willimantic → Windham |
| Blank measure | 0 |
| Thousands separator | Removed before integer conversion |
| Unit identifier | Registry `CTSP###00` state-publication unit; not FBI ORI |
| Duplicate key | Offense Type + Incident Date + Jurisdiction by Geography; prohibited |
| Required registry source coverage | CSP and bare-town rows for 79 / 79 towns × 7 offenses |

## Offense mapping

| Crime Insight member | Pipeline offense | Treatment |
|---|---|---|
| Murder and Nonnegligent Manslaughter | murder | Direct member |
| All Rape | rape | Precomputed source rollup |
| Robbery | robbery | Direct member |
| Aggravated Assault | aggravated_assault | Direct member |
| Burglary/Breaking & Entering | burglary | Direct member |
| Larceny_Theft Offenses Total | larceny | Precomputed source rollup of NIBRS 23A-23H |
| Motor Vehicle Theft | motor_vehicle_theft | Direct member |

## Rape exclusions

| Export member | Treatment |
|---|---|
| All Rape | Selected |
| Rape (2023 revised) | Not added to All Rape |
| Criminal Sexual Contact | Excluded |
| Incest | Excluded |
| Statutory Rape | Excluded |

## Larceny members

| Export rule | Value |
|---|---|
| Selected row | `Larceny_Theft Offenses Total` |
| Constituent rows in export | Not present |
| Parser arithmetic | Consume the source total once |
| Covered hierarchy | NIBRS 23A-23H |

| NIBRS code | Member | Treatment |
|---|---|---|
| 23A | Pocket-picking | Included in source total |
| 23B | Purse-snatching | Included in source total |
| 23C | Shoplifting | Included in source total |
| 23D | Theft From Building | Included in source total |
| 23E | Theft From Coin-Operated Machine or Device | Included in source total |
| 23F | Theft From Motor Vehicle | Included in source total |
| 23G | Theft of Motor Vehicle Parts or Accessories | Included in source total |
| 23H | All Other Larceny | Included in source total |

| Excluded member | Reason |
|---|---|
| Motor Vehicle Theft | Separate Part-I offense |
| Stolen Property Offenses | Receiving/possessing stolen property; not NIBRS 23A-23H larceny |
| Robbery | Separate Part-I offense |
| Burglary/Breaking & Entering | Separate Part-I offense |

## 2024 statewide comparison

| Measure | Count |
|---|---:|
| Crime Insight bare-town sum, selected seven offenses | 56,227 |
| DESPP SRS supplement, Index Crimes | 56,429 |
| Live export minus fixed supplement | -202 |
| Percent difference | -0.358% |

| Comparison note | Value |
|---|---|
| Parser hierarchy handling | No hierarchy rule re-applied; precomputed Connecticut_SS members consumed |
| Offense-level supplement reconciliation | Not available from the retained source artifacts |
| Aggregate interpretation | Live Crime Insight export vintage differs from the fixed 2024 SRS publication by 202 crimes |
