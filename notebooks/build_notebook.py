"""
Generates notebooks/nectar_analysis.ipynb - a walkthrough notebook that
calls into the src/ modules rather than duplicating logic, so the notebook
and the scripts never drift out of sync.
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells.append(md("""# Nectar Intelligent Facilities Platform — Analysis Walkthrough

Data Scientist Challenge submission. This notebook walks through all five
tasks using the reusable modules in `src/`, so the logic here matches what
`run_all.py` runs end-to-end. See `README.md` for setup and `reports/` for
the write-up.

Run `python src/generate_data.py` once before this notebook if `data/raw/`
is empty."""))

cells.append(code("""import sys
sys.path.insert(0, '../src')

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

pd.set_option('display.max_columns', 30)
%matplotlib inline"""))

cells.append(md("## Task 1 — Exploratory Data Analysis"))
cells.append(code("""import eda
telemetry, assets, connectivity = eda.load_raw()
print(f"{len(telemetry):,} telemetry rows across {telemetry.asset_id.nunique()} assets")
telemetry.head()"""))

cells.append(code("""miss = eda.missing_value_report(telemetry)
miss"""))

cells.append(code("""eda.run()  # regenerates all EDA plots + outputs/eda_summary.txt"""))

cells.append(code("""display(Image('../outputs/plots/02_daily_pattern.png'))
display(Image('../outputs/plots/06_failure_precursors.png'))
display(Image('../outputs/plots/07_correlation_heatmap.png'))"""))

cells.append(md("""**Key EDA findings**
- Power draw and occupancy both peak sharply during business hours (8am-7pm weekdays) and drop to baseline overnight/weekends — HVAC load is occupancy-driven, not constant.
- Vibration and power consumption jump noticeably (roughly 60% and 175% above baseline respectively) in the 24h window before a logged fault — a strong, learnable precursor signal.
- Pressure and vibration are strongly negatively correlated (~-0.85), consistent with pressure loss being a real mechanical symptom of the same wear that drives vibration up.
- Chillers run the hottest average power draw and the highest anomaly rate of any asset type — they're the highest-impact equipment to monitor closely."""))

cells.append(md("## Task 2 — Predictive Maintenance"))
cells.append(code("""import predictive_maintenance as pm
model, metrics, importances = pm.train()
metrics"""))

cells.append(code("""importances.head(15)"""))

cells.append(code("""display(Image('../outputs/plots/08_pm_feature_importance.png'))
display(Image('../outputs/plots/09_pm_pr_curve.png'))"""))

cells.append(md("""**Why XGBoost:** handles the mixed tabular features well, has built-in
imbalance handling (`scale_pos_weight`), and is fast enough to retrain daily
at this data volume.

**Top features:** rolling vibration volatility (16h and 96h std) dominates —
matches the EDA finding that vibration instability, not just its level, is
the clearest early warning sign. Power consumption trend and asset age follow.

**Business impact:** at the chosen threshold the model catches ~61% of
failures roughly a day ahead, at ~58% precision — meaning just over half of
maintenance alerts are real, which is a reasonable trade-off for a first
version given false negatives (missed failures) tend to cost far more than
an unnecessary inspection."""))

cells.append(md("## Task 3 — Energy Consumption Forecasting"))
cells.append(code("""import forecasting
fmodel, fmetrics = forecasting.train_and_evaluate()
fmetrics"""))

cells.append(code("""display(Image('../outputs/plots/10_forecast_vs_actual.png'))"""))

cells.append(md("""**Key insight:** day-of-week and the same-hour-last-week lag are the two most
important features — building energy demand is dominated by weekly occupancy
patterns (weekday office hours vs weekend) more than by short-term trends.
MAPE of ~11% is solid for a 24h-ahead building-level forecast and is good
enough to support proactive demand-response/optimization scheduling."""))

cells.append(md("## Task 4 — Anomaly Detection"))
cells.append(code("""import anomaly_detection as ad
adf, asummary = ad.run()
asummary"""))

cells.append(code("""display(Image('../outputs/plots/11_anomaly_timeline.png'))
display(Image('../outputs/plots/12_anomaly_by_asset_type.png'))"""))

cells.append(md("""**Methodology:** rolling z-score thresholding (fast, explainable, catches
single-sensor spikes) combined with Isolation Forest on the multivariate
sensor set (catches subtler combined-signal drift). Rows flagged by both are
treated as highest-confidence.

**Validation:** anomaly rate is roughly 6x higher on days an asset actually
faulted (8.3%) versus normal operation (1.4%) — a reasonable sanity check
that the flags correlate with real problems rather than just noise.

**Business recommendation:** route "flagged by both methods" alerts to
immediate inspection queues, and single-method flags to a lower-priority
watchlist, to keep alert fatigue manageable."""))

cells.append(md("## Task 5 — Multi-Asset Connectivity Analysis"))
cells.append(code("""import connectivity_analysis as ca
G, dq, impact_df = ca.run()"""))

cells.append(code("""display(Image('../outputs/plots/13_asset_hierarchy.png'))"""))

cells.append(code("""impact_df"""))

cells.append(md("""**Data quality issues found:** 2 duplicate connectivity edges and 2 records
referencing a parent asset that doesn't exist in the metadata table — both
the kind of quiet data issues that would silently break downstream impact
queries if not caught.

**Failure impact:** every chiller sits upstream of 8-10 downstream assets
(AHUs, pumps, sensors, meters) — a chiller failure is the highest blast-radius
event in the hierarchy, which matches operational intuition and supports
prioritizing chiller health monitoring above other asset types."""))

cells.append(md("## Summary\n\nAll five tasks are implemented as standalone, reusable modules in `src/`, sharing a common preprocessing pipeline. See `README.md` for the full write-up of assumptions and design trade-offs, and `reports/` for the condensed report."))

nb['cells'] = cells

out_path = Path(__file__).resolve().parent / "nectar_analysis.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)

print("wrote", out_path)
