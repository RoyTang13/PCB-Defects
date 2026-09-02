import streamlit as st
from config import SMOKE_SETTINGS
from utils.yolo_utils import uploaded_to_bgr, detect, prepare_processed_dataset, prepare_smoke_dataset, render_training_output, run_training_with_progress
from utils.preprocessing import nlm_edge_contour, nlm_edge_contour_smoke
st.title("Natasha — Non-Local Means + Edge / Contour")
st.caption("NLM reduces noise while preserving edges and fine details. Original → NLM → Canny/contours → YOLOv8n")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
if mode == "Experiment Mode":
    st.info("The smoke test scales images to 640 pixels wide before NLM so it completes quickly. This scale-only resize preserves normalized YOLO labels. The full 100-epoch experiment retains the original image dimensions.")
    if st.button("Run small Natasha smoke test"):
        run_training_with_progress("natasha_smoke_balanced", lambda progress: prepare_smoke_dataset("natasha_smoke_balanced", lambda x: nlm_edge_contour_smoke(x)[2], progress), SMOKE_SETTINGS["epochs"], "Natasha smoke test")
    if st.button("Prepare dataset and train Natasha (100 epochs)"):
        run_training_with_progress("natasha", lambda progress: prepare_processed_dataset("natasha", lambda x: nlm_edge_contour(x)[2], progress), 100, "Natasha full experiment")
    render_training_output("natasha_smoke_balanced")
    render_training_output("natasha")
else:
    t1, t2 = st.slider("Canny thresholds", 0, 255, (80, 160)); file = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png", "bmp"])
    if file:
        image = uploaded_to_bgr(file); den, edge, contour = nlm_edge_contour(image, t1, t2)
        for col, pic, label in zip(st.columns(4), (image, den, edge, contour), ("Original", "Denoised", "Edges", "Contours")): col.image(pic, channels="BGR" if pic.ndim == 3 else "GRAY", caption=label)
        if st.button("Run YOLOv8 Detection"):
            try:
                result, objects, _ = detect(contour, "natasha"); st.image(result, channels="BGR"); st.write(f"Defects: {len(objects)}")
            except Exception as e: st.warning(str(e))
