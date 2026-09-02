import streamlit as st
from config import SMOKE_SETTINGS
from utils.yolo_utils import uploaded_to_bgr, detect, prepare_processed_dataset, prepare_smoke_dataset, render_training_output, run_training_with_progress
from utils.preprocessing import gaussian_colour
st.title("Leyi — Gaussian Filtering + Colour Segmentation")
st.caption("Original → Gaussian noise reduction → HSV colour segmentation → YOLOv8n")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
if mode == "Experiment Mode":
    st.info("The fixed split is copied and processed without geometry changes, so its YOLO labels are reused unchanged.")
    if st.button("Run small Leyi smoke test"):
        run_training_with_progress("leyi_smoke_balanced", lambda progress: prepare_smoke_dataset("leyi_smoke_balanced", lambda x: gaussian_colour(x)[2], progress), SMOKE_SETTINGS["epochs"], "Leyi smoke test")
    if st.button("Prepare dataset and train Leyi (100 epochs)"):
        run_training_with_progress("leyi", lambda progress: prepare_processed_dataset("leyi", lambda x: gaussian_colour(x)[2], progress), 100, "Leyi full experiment")
    render_training_output("leyi_smoke_balanced")
    render_training_output("leyi")
else:
    k = st.select_slider("Gaussian kernel", [3, 5, 7], value=5); cols = st.columns(2)
    lo = tuple(cols[0].slider(x, 0, 179 if x == "H minimum" else 255, 0) for x in ("H minimum", "S minimum", "V minimum"))
    hi = tuple(cols[1].slider(x, 0, 179 if x == "H maximum" else 255, 179 if x == "H maximum" else 255) for x in ("H maximum", "S maximum", "V maximum"))
    file = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png", "bmp"])
    if file:
        image = uploaded_to_bgr(file); blurred, mask, segmented = gaussian_colour(image, k, lo, hi)
        for col, pic, label in zip(st.columns(4), (image, blurred, mask, segmented), ("Original", "Gaussian filtered", "HSV mask", "Segmented")): col.image(pic, channels="BGR" if pic.ndim == 3 else "GRAY", caption=label)
        if st.button("Run YOLOv8 Detection"):
            try:
                result, objects, _ = detect(segmented, "leyi"); st.image(result, channels="BGR"); st.write(f"Defects: {len(objects)}"); st.dataframe(objects, column_config={0:"Defect",1:"Confidence"})
            except Exception as e: st.warning(str(e))
