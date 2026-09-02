import streamlit as st
from config import SMOKE_SETTINGS
from utils.yolo_utils import uploaded_to_bgr, detect, prepare_manas_dataset, prepare_manas_smoke_dataset, render_training_output, run_training_with_progress
from utils.preprocessing import subtraction_morphology
st.title("Manas — Subtraction + Morphological Processing")
st.caption("Reference + defective → ORB alignment → subtraction → threshold → morphology → YOLOv8n")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
if mode == "Experiment Mode":
    st.info("Place same-named normal references in `dataset/reference_images/train`, `val`, and `test`. The fixed split and its labels are retained.")
    st.caption(f"Smoke test: {SMOKE_SETTINGS['train_per_class']}/{SMOKE_SETTINGS['val_per_class']}/{SMOKE_SETTINGS['test_per_class']} images per class, {SMOKE_SETTINGS['epochs']} epochs.")
    if st.button("Run small Manas smoke test"):
        run_training_with_progress("manas_smoke_balanced", lambda progress: prepare_manas_smoke_dataset(on_progress=progress), SMOKE_SETTINGS["epochs"], "Manas smoke test")
    if st.button("Prepare paired dataset and train Manas (100 epochs)"):
        run_training_with_progress("manas", lambda progress: prepare_manas_dataset(on_progress=progress), 100, "Manas full experiment")
    render_training_output("manas_smoke_balanced")
    render_training_output("manas")
else:
    kernel = st.select_slider("Morphology kernel size", [3, 5, 7, 9], value=5); iterations = st.slider("Iterations", 1, 5, 1)
    reference_file = st.file_uploader("Reference / normal PCB", type=["jpg", "jpeg", "png", "bmp"]); defective_file = st.file_uploader("Defective PCB", type=["jpg", "jpeg", "png", "bmp"])
    if reference_file and defective_file:
        ref, defect = uploaded_to_bgr(reference_file), uploaded_to_bgr(defective_file); aligned, diff, binary, morph, ok, msg = subtraction_morphology(ref, defect, kernel, iterations)
        (st.success if ok else st.warning)(msg)
        for col, pic, label in zip(st.columns(3), (ref, defect, aligned, diff, binary, morph), ("Reference", "Defective", "Aligned reference", "Difference", "Binary", "Morphological result")): col.image(pic, channels="BGR" if pic.ndim == 3 else "GRAY", caption=label)
        if st.button("Run YOLOv8 Detection"):
            try:
                result, objects, _ = detect(defect, "manas"); st.image(result, channels="BGR"); st.write(f"Defects: {len(objects)}")
            except Exception as e: st.warning(str(e))
