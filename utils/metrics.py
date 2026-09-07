from pathlib import Path
import pandas as pd
from config import RESULTS_DIR, EXPERIMENTS, FULL_RESULT_FOLDERS

def latest_result_paths():
    """Return completed full-experiment YOLO result files only.

    Only outputs registered as completed 100-epoch experiments are included.
    """
    found = {}
    for key in EXPERIMENTS:
        run_name = FULL_RESULT_FOLDERS[key]
        path = RESULTS_DIR / run_name / "results.csv"
        if path.exists():
            found[key] = (run_name, path)
    return found

def experiment_metrics():
    rows = []
    latest = latest_result_paths()
    for key, name in EXPERIMENTS.items():
        values = {"Precision": None, "Recall": None, "mAP50": None, "mAP50-95": None}
        run_name, csv_path = latest.get(key, (None, None))
        best_epoch = None
        if csv_path:
            frame = pd.read_csv(csv_path)
            # Ultralytics selects best.pt using validation fitness, which for
            # detection is governed by mAP50-95. Use that same best epoch.
            best_index = frame["metrics/mAP50-95(B)"].idxmax()
            best = frame.loc[best_index]
            best_epoch = int(best["epoch"]) + 1
            for label, column in (("Precision", "metrics/precision(B)"), ("Recall", "metrics/recall(B)"), ("mAP50", "metrics/mAP50(B)"), ("mAP50-95", "metrics/mAP50-95(B)")):
                values[label] = best.get(column)
        source = "Full experiment" if run_name else "N/A - experiment has not been run."
        rows.append({"Experiment": name, "Latest output": source, "Result folder": run_name, "Best epoch": best_epoch, **values})
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame.Experiment == EXPERIMENTS["baseline"], "mAP50"].iloc[0]
    baseline_full = frame.loc[frame.Experiment == EXPERIMENTS["baseline"], "Latest output"].iloc[0] == "Full experiment"
    frame["mAP50 improvement (%)"] = frame.apply(lambda row: None if not baseline_full or row["Latest output"] != "Full experiment" or pd.isna(row.mAP50) or pd.isna(baseline) or baseline == 0 else (row.mAP50-baseline)/baseline*100, axis=1)
    return frame
