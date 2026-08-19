"""
run_all.py

Runs the whole pipeline top to bottom, in order. Useful for a fresh checkout
or for re-running everything after changing the generator.

    python run_all.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("Generating synthetic IoT data", "src/generate_data.py"),
    ("Task 1 - EDA", "src/eda.py"),
    ("Task 2 - Predictive maintenance", "src/predictive_maintenance.py"),
    ("Task 3 - Energy forecasting", "src/forecasting.py"),
    ("Task 4 - Anomaly detection", "src/anomaly_detection.py"),
    ("Task 5 - Connectivity analysis", "src/connectivity_analysis.py"),
]

for label, script in STEPS:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    result = subprocess.run([sys.executable, script], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n{script} failed, stopping here.")
        sys.exit(1)

print("\nAll done. Check outputs/ for plots, metrics and reports, and models/ for the trained model.")
