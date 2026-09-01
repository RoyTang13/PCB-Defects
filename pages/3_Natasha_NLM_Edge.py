import streamlit as st
from utils.yolo_utils import uploaded_to_bgr, detect, train_experiment, prepare_processed_dataset
from utils.preprocessing import nlm_edge_contour
st.title("Natasha — Non-Local Means + Edge / Contour")
st.caption("NLM reduces noise while preserving edges and fine details. Original → NLM → Canny/contours → YOLOv8n")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
if mode == "Experiment Mode":
    if st.button("Prepare dataset and train Natasha (100 epochs)"):
        try: train_experiment("natasha", prepare_processed_dataset("natasha", lambda x: nlm_edge_contour(x)[2]))
        except Exception as e: st.error(str(e))
else:
    t1, t2 = st.slider("Canny thresholds", 0, 255, (80, 160)); file = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png", "bmp"])
    if file:
        image = uploaded_to_bgr(file); den, edge, contour = nlm_edge_contour(image, t1, t2)
        for col, pic, label in zip(st.columns(4), (image, den, edge, contour), ("Original", "Denoised", "Edges", "Contours")): col.image(pic, channels="BGR" if pic.ndim == 3 else "GRAY", caption=label)
        if st.button("Run YOLOv8 Detection"):
            result, objects, _ = detect(contour, "natasha"); st.image(result, channels="BGR"); st.write(f"Defects: {len(objects)}")
