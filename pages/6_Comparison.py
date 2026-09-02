import streamlit as st
from config import RESULTS_DIR
from utils.metrics import experiment_metrics, latest_result_paths
st.title("Final Experiment Comparison")
st.caption("The table selects the most recently completed real YOLO output for each technique. Smoke tests are labelled and must not be used as final scientific comparisons.")
frame = experiment_metrics(); display = frame.fillna("N/A - experiment has not been run.")
st.dataframe(display, hide_index=True)
if frame["Latest output"].astype(str).str.startswith("Smoke test").any():
    st.warning("Some latest outputs are smoke tests. Run all five full 100-epoch experiments before interpreting improvement percentages.")
for metric in ("Precision", "Recall", "mAP50", "mAP50-95"):
    available = frame.dropna(subset=[metric])
    st.subheader(f"{metric} comparison")
    if available.empty: st.info("N/A - experiment has not been run.")
    else: st.bar_chart(available, x="Experiment", y=metric)
st.subheader("Confusion matrices")
found = False
for key, (run_name, _) in latest_result_paths().items():
    path = RESULTS_DIR / run_name / "confusion_matrix.png"
    if path.exists(): st.image(str(path), caption=f"{key}: {run_name}"); found = True
if not found: st.info("N/A - experiment has not been run.")
