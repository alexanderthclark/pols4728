# ATUS recreation training data

Training data for the ATUS prediction competition: 136,146 respondents from
2003–2025, with one row per `TUCASEID`. Predictors describe the morning and
earlier respondent characteristics; the outcome records qualifying solo
recreation later in the diary day.

## Files

| File | Rows | Columns | Contents |
| --- | ---: | ---: | --- |
| [atus_train.parquet](atus_train.parquet) | 136,146 | 462 | Identifier, label, and default predictors. |
| [atus_train.csv.zip](atus_train.csv.zip) | 136,146 | 462 | ZIP archive containing `atus_train.csv`, with the same columns, values, and row order as the Parquet file. |
| [atus_train_labels.csv](atus_train_labels.csv) | 136,146 | 2 | Identifier and label only, for assembling predictors from BLS files. Same cases and row order as the full files. |

The CSV files have a header row and no row-index column. CSV does not store
column types; the types below describe the Parquet file. Preserve `TUCASEID`
as a 14-character string when reading any format or merging additional data.

## Full training-file schema

This schema applies to both `atus_train.parquet` and the CSV inside
`atus_train.csv.zip`. The 431 activity columns, four diary-calendar columns,
and 25 prior-CPS columns provide 460 default predictors.

| Column | Type | Meaning |
| --- | --- | --- |
| `TUCASEID` | string | Unique 14-digit ATUS case identifier; use as a merge key, not a predictor. |
| `y` | int8 | Binary outcome: `1` for a qualifying solo-recreation episode beginning at or after noon and before 4 a.m. the next morning; `0` otherwise. |
| `am_tNNNNNN` (431 columns) | int16 | Minutes spent in six-digit BLS activity code `NNNNNN` between 4 a.m. and noon. Episodes crossing noon contribute only minutes elapsed by noon. Values range from 0 to 480; these columns sum to 480 for every row. |
| `TUYEAR` | int32 | Year of the diary day. |
| `TUDIARYDAY` | int32 | Day of the week of the diary day, using BLS codes. |
| `TRHOLIDAY` | int32 | BLS flag indicating whether the diary day was a holiday. |
| `TUMONTH` | int32 | Month of the diary day. |
| `cps_*` (25 columns) | int32 | Variables from the respondent's earlier CPS interview, listed below. |

ATUS diaries cover 4 a.m. through 4 a.m. the next morning. An episode already
underway at noon does not count toward `y`; a later qualifying episode can.
The recreation basket includes sports and recreation (`1301xx`, excluding
routine fitness codes `130101`, `130124`, `130128`, `130131`, `130133`,
`130134`, and `130136`), spectator sports (`1302xx`), and performing arts,
museums, and movies (`120401`, `120402`, and `120403`). The qualifying episode
must be reported as done alone. General games (`120307`) are not included.
This is an outcome on one diary day, not a measure of loneliness.

### Prior CPS variables

The `cps_` prefix identifies the original BLS variable from the respondent's
ATUS-CPS record (`TULINENO = 1`). This interview preceded ATUS by 2–5 months.
All 25 columns are stored as `int32`; their values retain BLS codes.

| Column | Meaning |
| --- | --- |
| `cps_PRTAGE` | Age. |
| `cps_PESEX` | Sex. |
| `cps_PEEDUCA` | Highest level of education or degree completed. |
| `cps_PEMARITL` | Marital status. |
| `cps_PTDTRACE` | Race, using the BLS topcoded variable. |
| `cps_PRCITSHP` | Citizenship status. |
| `cps_GEREG` | Census region. |
| `cps_GESTFIPS` | State FIPS code. |
| `cps_HRNUMHOU` | Number of household members. |
| `cps_PRNMCHLD` | Number of own children under age 18. |
| `cps_HETENURE` | Housing tenure: owned, rented, or occupied without cash rent. |
| `cps_HEHOUSUT` | Type of housing unit. |
| `cps_HRHTYPE` | Household type. |
| `cps_PEMLR` | Monthly labor-force status recode. |
| `cps_PEHRUSLT` | Usual weekly hours worked across jobs. |
| `cps_PEIO1COW` | Class of worker in the main job. |
| `cps_PRMJIND1` | Major industry recode for the main job. |
| `cps_PRMJOCC1` | Major occupation recode for the main job. |
| `cps_PRWKSTAT` | Full-time or part-time work status. |
| `cps_PESCHENR` | Enrollment in high school, college, or university in the prior week. |
| `cps_PESCHFT` | Full-time or part-time school enrollment. |
| `cps_HEFAMINC` | Family income category, newer field. |
| `cps_HUFAMINC` | Family income category, older field. |
| `cps_HRMONTH` | Month of the earlier CPS interview, not the diary day. |
| `cps_HRYEAR4` | Year of the earlier CPS interview, not the diary day. |

Integer storage does not make categorical codes continuous quantities.
Negative values are BLS special or missing-response codes; consult the
dictionaries below for their meanings. The family-income source field
changed in January 2010, so both versions are retained. An unavailable income
field does not mean zero income.

## Labels-only schema

`atus_train_labels.csv` contains exactly these two columns:

| Column | Logical type | Meaning |
| --- | --- | --- |
| `TUCASEID` | string | Same unique 14-digit case identifier as in the full training files. |
| `y` | binary integer | Same `0`/`1` outcome as in the full training files. |

## BLS documentation

The source files and detailed category definitions are available from BLS:

- [ATUS 2003–2025 multi-year data files](https://www.bls.gov/tus/data/datafiles-0325.htm)
- [ATUS interview data dictionary](https://www.bls.gov/tus/dictionaries/atusintcodebk0325.pdf) for diary-calendar variables
- [ATUS-CPS data dictionary](https://www.bls.gov/tus/dictionaries/atuscpscodebk0325.pdf) for the original variables underlying `cps_*`
- [ATUS activity coding lexicon](https://www.bls.gov/tus/lexicons/lexiconnoex0325.pdf) for the six-digit codes in `am_tNNNNNN`

Additional predictors must follow the homework's noon cutoff and source
restrictions. Full-day time-use totals include behavior after the prediction
time and are not interchangeable with the supplied morning totals.
