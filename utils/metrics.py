from pathlib import Path
import pandas as pd
from config import RESULTS_DIR, EXPERIMENTS

def experiment_metrics():
    rows = []
    for key, name in EXPERIMENTS.items():
        csv_path = RESULTS_DIR / key / "results.csv"
        values = {"Precision": None, "Recall": None, "mAP50": None, "mAP50-95": None}
        if csv_path.exists():
            frame = pd.read_csv(csv_path); last = frame.iloc[-1]
            for label, column in (("Precision", "metrics/precision(B)"), ("Recall", "metrics/recall(B)"), ("mAP50", "metrics/mAP50(B)"), ("mAP50-95", "metrics/mAP50-95(B)")):
                values[label] = last.get(column)
        rows.append({"Experiment": name, **values})
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame.Experiment == EXPERIMENTS["baseline"], "mAP50"].iloc[0]
    frame["mAP50 improvement (%)"] = frame.mAP50.apply(lambda v: None if pd.isna(v) or pd.isna(baseline) or baseline == 0 else (v-baseline)/baseline*100)
    return frame
