import streamlit as st
import matplotlib.pyplot as plt
from config import RESULTS_DIR
from utils.metrics import experiment_metrics
st.title("Final Experiment Comparison")
st.caption("Metrics are loaded only from actual Ultralytics training results; unavailable experiments remain N/A.")
frame = experiment_metrics(); display = frame.fillna("N/A - experiment has not been run.")
st.dataframe(display, use_container_width=True, hide_index=True)
for metric in ("Precision", "Recall", "mAP50", "mAP50-95"):
    available = frame.dropna(subset=[metric])
    st.subheader(f"{metric} comparison")
    if available.empty: st.info("N/A - experiment has not been run.")
    else:
        fig, ax = plt.subplots(); ax.bar(available.Experiment, available[metric]); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=25); st.pyplot(fig)
st.subheader("Confusion matrices")
found = False
for key in ("baseline", "leyi", "natasha", "jiading", "manas"):
    path = RESULTS_DIR / key / "confusion_matrix.png"
    if path.exists(): st.image(str(path), caption=key); found = True
if not found: st.info("N/A - experiment has not been run.")
