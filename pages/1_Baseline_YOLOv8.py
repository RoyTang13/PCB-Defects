import streamlit as st
from utils.yolo_utils import uploaded_to_bgr, detect, train_experiment
st.title("Baseline YOLOv8 — Control Experiment")
st.caption("Original PCB image → YOLOv8n → defect detection. No custom preprocessing is applied.")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
if mode == "Experiment Mode":
    if st.button("Train baseline (100 epochs)"): train_experiment("baseline")
else:
    file = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png", "bmp"])
    if file:
        image = uploaded_to_bgr(file); st.image(image, channels="BGR", caption="Original image")
        if st.button("Run YOLOv8 Detection"):
            try:
                result, objects, ms = detect(image, "baseline"); st.image(result, channels="BGR", caption="YOLO detection")
                st.write(f"Defects detected: {len(objects)} | Inference: {ms:.1f} ms" if ms else f"Defects detected: {len(objects)}")
                st.dataframe({"Defect": [x[0] for x in objects], "Confidence": [round(x[1], 3) for x in objects]})
            except Exception as e: st.error(str(e))
