# Baseline YOLOv8 control experiment.
import streamlit as st
from utils.yolo_utils import uploaded_to_bgr, detect, train_experiment, render_training_output
st.title("Baseline YOLOv8 — Control Experiment")
st.caption("Original PCB image → YOLOv8n → defect detection. No custom preprocessing is applied.")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
if mode == "Experiment Mode":
    # Train the unprocessed baseline model.
    if st.button("Train baseline (full dataset, 100 epochs)"):
        try: train_experiment("baseline")
        except Exception as e: st.error(str(e))
    render_training_output("baseline")
else:
    # Upload an image for direct YOLO detection.
    file = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png", "bmp"])
    if file:
        image = uploaded_to_bgr(file); st.image(image, channels="BGR", caption="Original image")
        if st.button("Run YOLOv8 Detection"):
            # Run detection and display the detected defects.
            try:
                result, objects, ms = detect(image, "baseline"); st.image(result, channels="BGR", caption="YOLO detection")
                st.write(f"Defects detected: {len(objects)} | Inference: {ms:.1f} ms" if ms else f"Defects detected: {len(objects)}")
                st.dataframe({"Defect": [x[0] for x in objects], "Confidence": [round(x[1], 3) for x in objects]})
            except Exception as e: st.error(str(e))
