from pathlib import Path
import pandas as pd
from config import RESULTS_DIR, EXPERIMENTS

def latest_result_paths():
    """Return the newest real YOLO results.csv for each experiment, including smoke tests."""
    found = {}
    for key in EXPERIMENTS:
        names = [key, "smoke_balanced", "smoke_test"] if key == "baseline" else [key, f"{key}_smoke_balanced", f"{key}_smoke_test"]
        candidates = [(name, RESULTS_DIR / name / "results.csv") for name in names]
        available = [(name, path) for name, path in candidates if path.exists()]
        if available:
            found[key] = max(available, key=lambda item: item[1].stat().st_mtime)
    return found

def experiment_metrics():
    rows = []
    latest = latest_result_paths()
    for key, name in EXPERIMENTS.items():
        values = {"Precision": None, "Recall": None, "mAP50": None, "mAP50-95": None}
        run_name, csv_path = latest.get(key, (None, None))
        if csv_path:
            frame = pd.read_csv(csv_path); last = frame.iloc[-1]
            for label, column in (("Precision", "metrics/precision(B)"), ("Recall", "metrics/recall(B)"), ("mAP50", "metrics/mAP50(B)"), ("mAP50-95", "metrics/mAP50-95(B)")):
                values[label] = last.get(column)
        source = "Full experiment" if run_name == key else (f"Smoke test ({run_name})" if run_name else "N/A - experiment has not been run.")
        rows.append({"Experiment": name, "Latest output": source, "Result folder": run_name, **values})
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame.Experiment == EXPERIMENTS["baseline"], "mAP50"].iloc[0]
    baseline_full = frame.loc[frame.Experiment == EXPERIMENTS["baseline"], "Latest output"].iloc[0] == "Full experiment"
    frame["mAP50 improvement (%)"] = frame.apply(lambda row: None if not baseline_full or row["Latest output"] != "Full experiment" or pd.isna(row.mAP50) or pd.isna(baseline) or baseline == 0 else (row.mAP50-baseline)/baseline*100, axis=1)
    return frame
