# Non-local means denoising and edge detection experiment.
import streamlit as st
from utils.yolo_utils import uploaded_to_bgr, detect, prepare_processed_dataset, render_training_output, run_training_with_progress
from utils.preprocessing import nlm_edge_contour
st.title("Natasha — Non-Local Means + Edge / Contour")
st.caption("NLM reduces noise while preserving edges and fine details. Original → NLM → Canny/contours → YOLOv8n")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
if mode == "Experiment Mode":
    # Prepare the edge-processed dataset and train the Natasha model.
    st.info("The fixed split is processed without geometry changes, so its YOLO labels are reused unchanged.")
    if st.button("Prepare dataset and train Natasha (100 epochs)"):
        run_training_with_progress("natasha", lambda progress: prepare_processed_dataset("natasha", lambda x: nlm_edge_contour(x)[2], progress), 100, "Natasha full experiment")
    render_training_output("natasha")
else:
    # Set Canny thresholds and upload an image for the demo.
    t1, t2 = st.slider("Canny thresholds", 0, 255, (80, 160)); file = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png", "bmp"])
    if file:
        # Apply NLM denoising, edge detection, and contour extraction.
        image = uploaded_to_bgr(file); den, edge, contour = nlm_edge_contour(image, t1, t2)
        for col, pic, label in zip(st.columns(4), (image, den, edge, contour), ("Original", "Denoised", "Edges", "Contours")): col.image(pic, channels="BGR" if pic.ndim == 3 else "GRAY", caption=label)
        if st.button("Run YOLOv8 Detection"):
            # Detect defects in the contour image.
            try:
                result, objects, _ = detect(contour, "natasha"); st.image(result, channels="BGR"); st.write(f"Defects: {len(objects)}")
            except Exception as e: st.warning(str(e))
