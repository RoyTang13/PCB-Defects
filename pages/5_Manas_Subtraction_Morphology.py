from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from utils.manas_preprocessing import find_candidate_regions, subtraction_morphology
from utils.yolo_utils import detect, prepare_manas_dataset, render_training_output, run_training_with_progress, uploaded_to_bgr


def board_id(filename):
    return Path(filename).stem.split("_", 1)[0].lower()


def show_images(images):
    for start in range(0, len(images), 3):
        row = st.columns(3)
        for column, (image, caption) in zip(row, images[start : start + 3]):
            column.image(image, channels="GRAY" if image.ndim == 2 else "BGR", caption=caption)


st.title("Manas — Subtraction + Morphological Processing")
st.caption("Reference + defective → ORB alignment → subtraction → Otsu thresholding → morphology → YOLOv8n")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
settings = st.columns(2)
kernel_size = settings[0].select_slider("Morphology kernel size", options=[3, 5, 7, 9], value=5)
iterations = settings[1].slider("Morphology iterations", 1, 5, 1)

if mode == "Experiment Mode":
    st.info("Normal PCB references must be stored in `dataset/reference_images`. The defective image and its reference must have the same board ID, for example `01.JPG` and `01_mouse_bite_07.jpg`.")
    st.caption(f"Current preprocessing settings: {kernel_size}×{kernel_size} kernel, {iterations} iteration(s). The fixed split and labels are retained.")
    if st.button("Prepare paired dataset and train Manas (100 epochs)"):
        run_training_with_progress("manas", lambda progress: prepare_manas_dataset(kernel_size=kernel_size, iterations=iterations, on_progress=progress), 100, "Manas full experiment")
    render_training_output("manas")
else:
    st.caption("Upload the original image files rather than screenshots of the images.")
    reference_file = st.file_uploader("Reference / normal PCB", type=["jpg", "jpeg", "png", "bmp"])
    defective_file = st.file_uploader("Defective PCB", type=["jpg", "jpeg", "png", "bmp"])
    if reference_file and defective_file:
        uploaded_names = (reference_file.name.lower(), defective_file.name.lower())
        if any(name.startswith("screen") for name in uploaded_names):
            st.warning("These files look like screenshots. Original PCB files give more reliable alignment.")
        reference_name, defective_name = board_id(reference_file.name), board_id(defective_file.name)
        if reference_name != defective_name:
            st.warning(f"The board IDs do not match ({reference_name} and {defective_name}). Use a normal and defective image of the same PCB.")
        reference, defective = uploaded_to_bgr(reference_file), uploaded_to_bgr(defective_file)
        aligned, difference, binary, morphology, aligned_ok, alignment_message = subtraction_morphology(reference, defective, kernel_size, iterations)
        (st.success if aligned_ok else st.warning)(alignment_message)
        show_images([(reference, "Reference"), (defective, "Defective"), (aligned, "Aligned reference"), (difference, "Absolute difference"), (binary, "Otsu binary mask"), (morphology, "After opening and closing")])
        candidate_regions = find_candidate_regions(morphology)
        white_pixel_percentage = np.count_nonzero(morphology) / morphology.size * 100
        summary = st.columns(2)
        summary[0].metric("Candidate regions", len(candidate_regions))
        summary[1].metric("White pixels", f"{white_pixel_percentage:.3f}%")
        if candidate_regions:
            with st.expander("Candidate-region measurements"):
                st.dataframe(candidate_regions, hide_index=True)
        if st.button("Run YOLOv8 Detection"):
            try:
                model_input = cv2.cvtColor(morphology, cv2.COLOR_GRAY2BGR)
                result, objects, inference_time = detect(model_input, "manas")
                st.image(result, channels="BGR", caption="YOLOv8 detection on morphology image")
                result_columns = st.columns(2)
                result_columns[0].metric("YOLO defects", len(objects))
                if inference_time is not None:
                    result_columns[1].metric("Inference time", f"{inference_time:.1f} ms")
                if objects:
                    st.dataframe([{"Defect": name, "Confidence": round(confidence, 3)} for name, confidence in objects], hide_index=True)
                else:
                    st.info("No YOLO defect was found. Candidate regions above come from classical image processing; YOLO results will remain unreliable until Manas training finishes.")
            except Exception as error:
                st.warning(str(error))
