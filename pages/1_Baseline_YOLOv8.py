import streamlit as st
from config import SMOKE_SETTINGS
from utils.yolo_utils import uploaded_to_bgr, detect, train_experiment, prepare_smoke_dataset, render_training_output
st.title("Baseline YOLOv8 — Control Experiment")
st.caption("Original PCB image → YOLOv8n → defect detection. No custom preprocessing is applied.")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
if mode == "Experiment Mode":
    st.subheader("Quick training check")
    st.caption(f"Uses all six classes: {SMOKE_SETTINGS['train_per_class']} train / {SMOKE_SETTINGS['val_per_class']} validation / {SMOKE_SETTINGS['test_per_class']} test images per class for {SMOKE_SETTINGS['epochs']} epochs. It is not a comparison result.")
    if st.button("Run small smoke test"):
        try:
            train_experiment("smoke_balanced", prepare_smoke_dataset("smoke_balanced"), epochs=SMOKE_SETTINGS["epochs"])
            st.success("Smoke test completed. Review results/smoke_balanced before starting the full experiment.")
        except Exception as e: st.error(str(e))
    st.divider()
    if st.button("Train baseline (full dataset, 100 epochs)"):
        try: train_experiment("baseline")
        except Exception as e: st.error(str(e))
    render_training_output("smoke_balanced")
    render_training_output("baseline")
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
