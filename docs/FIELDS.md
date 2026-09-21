# Published fields

Generated from `frontend/build/crschema.py` by `docs/make_fields.py`. Edition 2025.1.

Columns: 64 in the block-group file, 63 in the tract file.
The tract file omits `block_group_geoid`.

Support is the geography the value is estimated at.
Murder and rape rate and index columns are null in the block-group file.
Murder and rape block-group expected counts are within-tract allocations for reconciliation.

| Column | Meaning | Unit | Support |
|---|---|---|---|
| `block_group_geoid` | 2020 Census block group identifier, 12 digits. | Identifier | Block group |
| `tract_id` | 2020 Census tract identifier, 11 digits. | Identifier | Block group and tract |
| `state_fips` | State FIPS code, two digits. | Identifier | Block group and tract |
| `state` | State postal abbreviation, two letters. | Identifier | Block group and tract |
| `population_2025` | Resident population, 2025. | Persons | Block group and tract |
| `land_area_sq_mi` | Land area, 2020 TIGER, water excluded. | Square miles | Block group and tract |
| `special_use_type` | Area character: ordinary, park_open_space, industrial_employment, institutional_facility, campus_institution, transient_destination, group_quarters_other, unknown_special_use. | Category | Block group and tract |
| `jurisdiction_id` | Police jurisdiction whose total anchors this area. | Identifier | Block group and tract |
| `jurisdiction_name` | Display name of that police jurisdiction. | Text | Block group and tract |
| `murder_expected_count` | Estimated Murder offenses in 2025. | Offenses | Block group and tract |
| `murder_primary_denominator` | Exposure denominator for Murder: persons, premises or vehicles. | Exposure units | Block group and tract |
| `murder_rate_primary` | Murder offenses per 100,000 exposure units. | Per 100,000 | Tract |
| `murder_index_primary` | Murder Crime exposure index. | Index, 100 = US average | Tract |
| `murder_rate_resident` | Murder offenses per 100,000 residents. | Per 100,000 | Tract |
| `murder_index_resident` | Murder Per resident index. | Index, 100 = US average | Tract |
| `murder_source_mode` | Origin of the Murder estimate: direct_city_incident, mixed, modeled_transfer. | Category | Tract |
| `rape_expected_count` | Estimated Rape offenses in 2025. | Offenses | Block group and tract |
| `rape_primary_denominator` | Exposure denominator for Rape: persons, premises or vehicles. | Exposure units | Block group and tract |
| `rape_rate_primary` | Rape offenses per 100,000 exposure units. | Per 100,000 | Tract |
| `rape_index_primary` | Rape Crime exposure index. | Index, 100 = US average | Tract |
| `rape_rate_resident` | Rape offenses per 100,000 residents. | Per 100,000 | Tract |
| `rape_index_resident` | Rape Per resident index. | Index, 100 = US average | Tract |
| `rape_source_mode` | Origin of the Rape estimate: direct_city_incident, mixed, modeled_transfer. | Category | Tract |
| `robbery_expected_count` | Estimated Robbery offenses in 2025. | Offenses | Block group and tract |
| `robbery_primary_denominator` | Exposure denominator for Robbery: persons, premises or vehicles. | Exposure units | Block group and tract |
| `robbery_rate_primary` | Robbery offenses per 100,000 exposure units. | Per 100,000 | Block group and tract |
| `robbery_index_primary` | Robbery Crime exposure index. | Index, 100 = US average | Block group and tract |
| `robbery_rate_resident` | Robbery offenses per 100,000 residents. | Per 100,000 | Block group and tract |
| `robbery_index_resident` | Robbery Per resident index. | Index, 100 = US average | Block group and tract |
| `robbery_source_mode` | Origin of the Robbery estimate: direct_city_incident, mixed, modeled_transfer. | Category | Block group and tract |
| `aggravated_assault_expected_count` | Estimated Aggravated assault offenses in 2025. | Offenses | Block group and tract |
| `aggravated_assault_primary_denominator` | Exposure denominator for Aggravated assault: persons, premises or vehicles. | Exposure units | Block group and tract |
| `aggravated_assault_rate_primary` | Aggravated assault offenses per 100,000 exposure units. | Per 100,000 | Block group and tract |
| `aggravated_assault_index_primary` | Aggravated assault Crime exposure index. | Index, 100 = US average | Block group and tract |
| `aggravated_assault_rate_resident` | Aggravated assault offenses per 100,000 residents. | Per 100,000 | Block group and tract |
| `aggravated_assault_index_resident` | Aggravated assault Per resident index. | Index, 100 = US average | Block group and tract |
| `aggravated_assault_source_mode` | Origin of the Aggravated assault estimate: direct_city_incident, mixed, modeled_transfer. | Category | Block group and tract |
| `burglary_expected_count` | Estimated Burglary offenses in 2025. | Offenses | Block group and tract |
| `burglary_primary_denominator` | Exposure denominator for Burglary: persons, premises or vehicles. | Exposure units | Block group and tract |
| `burglary_rate_primary` | Burglary offenses per 100,000 exposure units. | Per 100,000 | Block group and tract |
| `burglary_index_primary` | Burglary Crime exposure index. | Index, 100 = US average | Block group and tract |
| `burglary_rate_resident` | Burglary offenses per 100,000 residents. | Per 100,000 | Block group and tract |
| `burglary_index_resident` | Burglary Per resident index. | Index, 100 = US average | Block group and tract |
| `burglary_source_mode` | Origin of the Burglary estimate: direct_city_incident, mixed, modeled_transfer. | Category | Block group and tract |
| `larceny_expected_count` | Estimated Larceny/theft offenses in 2025. | Offenses | Block group and tract |
| `larceny_primary_denominator` | Exposure denominator for Larceny/theft: persons, premises or vehicles. | Exposure units | Block group and tract |
| `larceny_rate_primary` | Larceny/theft offenses per 100,000 exposure units. | Per 100,000 | Block group and tract |
| `larceny_index_primary` | Larceny/theft Crime exposure index. | Index, 100 = US average | Block group and tract |
| `larceny_rate_resident` | Larceny/theft offenses per 100,000 residents. | Per 100,000 | Block group and tract |
| `larceny_index_resident` | Larceny/theft Per resident index. | Index, 100 = US average | Block group and tract |
| `larceny_source_mode` | Origin of the Larceny/theft estimate: direct_city_incident, mixed, modeled_transfer. | Category | Block group and tract |
| `motor_vehicle_theft_expected_count` | Estimated Motor vehicle theft offenses in 2025. | Offenses | Block group and tract |
| `motor_vehicle_theft_primary_denominator` | Exposure denominator for Motor vehicle theft: persons, premises or vehicles. | Exposure units | Block group and tract |
| `motor_vehicle_theft_rate_primary` | Motor vehicle theft offenses per 100,000 exposure units. | Per 100,000 | Block group and tract |
| `motor_vehicle_theft_index_primary` | Motor vehicle theft Crime exposure index. | Index, 100 = US average | Block group and tract |
| `motor_vehicle_theft_rate_resident` | Motor vehicle theft offenses per 100,000 residents. | Per 100,000 | Block group and tract |
| `motor_vehicle_theft_index_resident` | Motor vehicle theft Per resident index. | Index, 100 = US average | Block group and tract |
| `motor_vehicle_theft_source_mode` | Origin of the Motor vehicle theft estimate: direct_city_incident, mixed, modeled_transfer. | Category | Block group and tract |
| `overall_crime_exposure` | All seven offenses combined, exposure denominators. | Index, 100 = US average | Block group and tract |
| `overall_per_resident` | All seven offenses combined, resident denominator. | Index, 100 = US average | Block group and tract |
| `violent_crime_exposure` | Murder, rape, robbery and aggravated assault, exposure denominators. | Index, 100 = US average | Block group and tract |
| `violent_per_resident` | Murder, rape, robbery and aggravated assault, resident denominator. | Index, 100 = US average | Block group and tract |
| `property_crime_exposure` | Burglary, larceny and motor vehicle theft, exposure denominators. | Index, 100 = US average | Block group and tract |
| `property_per_resident` | Burglary, larceny and motor vehicle theft, resident denominator. | Index, 100 = US average | Block group and tract |
