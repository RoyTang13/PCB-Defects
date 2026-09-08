import cv2
import streamlit as st

from utils.manas_preprocessing import subtraction_morphology
from utils.preprocessing import gaussian_colour, mild_clahe_unsharp, nlm_edge_contour
from utils.yolo_utils import detect, uploaded_to_bgr


st.title("All preprocessing comparison")
st.caption(
    "Apply every preprocessing pipeline to one PCB defect image, then compare the resulting YOLO detections."
)
st.info(
    "Manas subtraction + morphology requires a matching normal/reference PCB image. "
    "The other three pipelines run from the defect image alone."
)

with st.form("all_preprocessing_form"):
    defect_file = st.file_uploader(
        "Defect PCB image",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    reference_file = st.file_uploader(
        "Matching normal/reference PCB image (required for Manas)",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    run_all = st.form_submit_button("Run all preprocessing pipelines")

if run_all:
    if defect_file is None:
        st.error("Upload a defect PCB image first.")
    else:
        defective = uploaded_to_bgr(defect_file)
        st.subheader("Uploaded defect image")
        st.image(defective, channels="BGR", caption="Original defect PCB image")

        pipelines = [
            (
                "Gaussian filtering + colour segmentation",
                "leyi",
                gaussian_colour(defective)[2],
                "Gaussian kernel 5 with the full HSV range used during training.",
            ),
            (
                "Non-local means + edge / contour",
                "natasha",
                nlm_edge_contour(defective)[2],
                "NLM denoising followed by Canny edges and contours.",
            ),
            (
                "Mild LAB-CLAHE + unsharp masking",
                "jiading_100epochs",
                mild_clahe_unsharp(defective),
                "Mild contrast enhancement with threshold-controlled sharpening.",
            ),
        ]

        if reference_file is not None:
            reference = uploaded_to_bgr(reference_file)
            _, _, _, morphology, aligned_ok, alignment_message = subtraction_morphology(reference, defective)
            highlighted = defective.copy()
            highlighted[morphology > 0] = (0, 0, 255)
            colour_input = cv2.addWeighted(defective, 0.80, highlighted, 0.20, 0)
            pipelines.append(
                (
                    "Subtraction + morphology",
                    "manas",
                    colour_input,
                    f"{alignment_message} Colour-preserved YOLO input.",
                )
            )
            if not aligned_ok:
                st.warning("Manas alignment used a fallback. Its result may be less reliable.")
        else:
            st.warning("Manas was skipped because no matching normal/reference image was uploaded.")

        for title, experiment, processed, detail in pipelines:
            with st.container(border=True):
                st.subheader(title)
                st.caption(detail)
                processed_column, detection_column = st.columns(2)
                processed_column.image(processed, channels="BGR", caption="After preprocessing")
                try:
                    detected, objects, inference_time = detect(processed, experiment)
                    detection_column.image(detected, channels="BGR", caption="YOLOv8 detection result")
                    inference_label = "N/A" if inference_time is None else f"{inference_time:.1f} ms"
                    metrics = detection_column.columns(2)
                    metrics[0].metric("Detected defects", len(objects))
                    metrics[1].metric("Inference", inference_label)
                    if objects:
                        detection_column.dataframe(
                            {
                                "Defect": [label for label, _ in objects],
                                "Confidence": [round(confidence, 3) for _, confidence in objects],
                            },
                            hide_index=True,
                        )
                    else:
                        detection_column.info("No defect reached the confidence threshold.")
                except Exception as error:
                    detection_column.error(f"Detection could not run: {error}")
