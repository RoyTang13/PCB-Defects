import streamlit as st
from utils.yolo_utils import uploaded_to_bgr, detect, train_experiment, prepare_manas_dataset
from utils.preprocessing import subtraction_morphology
st.title("Manas — Subtraction + Morphological Processing")
st.caption("Reference + defective → ORB alignment → subtraction → threshold → morphology → YOLOv8n")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
if mode == "Experiment Mode":
    st.info("Place same-named normal references in `dataset/reference_images/train`, `val`, and `test`. The fixed split and its labels are retained.")
    if st.button("Prepare paired dataset and train Manas (100 epochs)"):
        try: train_experiment("manas", prepare_manas_dataset())
        except Exception as e: st.error(str(e))
else:
    kernel = st.select_slider("Morphology kernel size", [3, 5, 7, 9], value=5); iterations = st.slider("Iterations", 1, 5, 1)
    reference_file = st.file_uploader("Reference / normal PCB", type=["jpg", "jpeg", "png", "bmp"]); defective_file = st.file_uploader("Defective PCB", type=["jpg", "jpeg", "png", "bmp"])
    if reference_file and defective_file:
        ref, defect = uploaded_to_bgr(reference_file), uploaded_to_bgr(defective_file); aligned, diff, binary, morph, ok, msg = subtraction_morphology(ref, defect, kernel, iterations)
        (st.success if ok else st.warning)(msg)
        for col, pic, label in zip(st.columns(3), (ref, defect, aligned, diff, binary, morph), ("Reference", "Defective", "Aligned reference", "Difference", "Binary", "Morphological result")): col.image(pic, channels="BGR" if pic.ndim == 3 else "GRAY", caption=label)
        if st.button("Run YOLOv8 Detection"):
            result, objects, _ = detect(defect, "manas"); st.image(result, channels="BGR"); st.write(f"Defects: {len(objects)}")
