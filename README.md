# OKR and Jira reporting

Turns the raw OKR and Jira exports into a management report and a data-quality log.

## Run it
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/build_report.py     # pipeline + report
.venv/bin/python -m pytest tests         # tests
```
Put new exports in `data/raw/`. The newest file matching each pattern in the settings is used.

## What is where
| Path | Contents |
|---|---|
| `reports/okr_report.html`, `.pdf` | The report. The HTML works offline with no install |
| `data/processed/data_quality_log.xlsx` | Data-quality log |
| `data/processed/*.csv` | Clean OKR and Jira tables, team bridge, reconciliation, grade comparison |
| `config/settings.yaml` | Business rules: thresholds, status groups, tier names, column names |
| `src/pipeline.py` | Cleaning, hierarchy, reconciliation, data-quality checks |
| `src/preparation.py` | Exploration evidence and the Data preparation tab: each step's finding, decision and effect |
| `src/build_report.py` | KPIs, charts, insights and the report page |
| `tests/` | Checks on the data, plus deliberately changed input files |

## When something changes
| Change | What happens | What to do |
|---|---|---|
| New export, or a new year | Runs normally; the report defaults to the export date's cycle | Nothing |
| Columns reordered or added | Runs normally | Nothing |
| Column renamed or missing | Run stops and names the column | Update the column map in the settings |
| New status or month name | Run stops and names the value | Add it to the settings |
| A few rows with broken links | Runs, flags them, and adds warnings to `run_summary.json` | Owners review |

If a run stops, nothing is overwritten, so the last good output remains in `data/processed/` and `reports/`.

## Main assumptions
- The grade scale is 0-100. This is not stated in the files and needs confirming with the OKR owner.
- `DEMO-n` pairs with `X-n`. Used only to repair shared issue keys, and only while it holds on at least 95% of issues.
- Export language can vary, so month names in other languages may need adding to the settings.

## Automation idea
The full version is in the report's Automation idea tab. In short:
- **Automate:** collecting both exports, cleaning, linking and checking them, and rebuilding the report and data-quality log.
- **Accuracy:** one rules file, checks that stop the run if a total would be wrong, and data-quality checks on every run.
- **If something goes wrong:** nothing is published and the last good report stays online. Small problems are flagged for owners without stopping the report.
- **Built here:** the rules file, the stopping checks, keeping the last good output, and the tests. **To add:** scheduling, automatic fetching, alerts and run history.
