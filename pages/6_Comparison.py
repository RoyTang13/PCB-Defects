import streamlit as st
from config import RESULTS_DIR
from utils.metrics import experiment_metrics, latest_result_paths
st.title("Final Experiment Comparison")
st.caption("The table reports each completed 100-epoch experiment at the validation epoch selected for its best.pt checkpoint.")
frame = experiment_metrics()

jia_ding = frame.loc[frame["Experiment"] == "Mild LAB-CLAHE + Unsharp Masking"]
baseline = frame.loc[frame["Experiment"] == "Baseline YOLOv8"]
if not jia_ding.empty and not baseline.empty:
    jia_ding, baseline = jia_ding.iloc[0], baseline.iloc[0]
    if all(not value is None for value in (jia_ding["Recall"], jia_ding["mAP50"], baseline["Recall"], baseline["mAP50"])):
        with st.container(border=True):
            st.subheader("Mild LAB-CLAHE + Unsharp Masking advantages over baseline")
            recall, map50 = st.columns(2)
            recall.metric(
                "Recall",
                f"{jia_ding['Recall']:.4f}",
                f"{(jia_ding['Recall'] - baseline['Recall']) * 100:+.2f} percentage points",
            )
            map50.metric(
                "mAP50",
                f"{jia_ding['mAP50']:.4f}",
                f"{(jia_ding['mAP50'] - baseline['mAP50']) * 100:+.2f} percentage points",
            )
            st.caption(
                f"Mild LAB-CLAHE + Unsharp Masking has a lower mAP50-95 ({jia_ding['mAP50-95']:.4f}) than "
                f"the baseline ({baseline['mAP50-95']:.4f}); the full table remains below."
            )

display = frame.fillna("N/A - experiment has not been run.")
st.dataframe(display, hide_index=True)
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
