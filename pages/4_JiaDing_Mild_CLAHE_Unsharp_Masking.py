import streamlit as st

from config import SMOKE_SETTINGS

from utils.yolo_utils import (
    uploaded_to_bgr,
    detect,
    prepare_processed_dataset,
    prepare_smoke_dataset,
    render_training_output,
    run_training_with_progress,
)

from utils.preprocessing import mild_clahe_unsharp


EXPERIMENT_NAME = "jiading_100epochs"
SMOKE_EXPERIMENT_NAME = "jiading_smoke_balanced"


st.title("Jia Ding — Mild LAB-CLAHE + Unsharp Masking")

st.caption(
    "Original → Mild LAB-CLAHE → "
    "Threshold-Controlled Unsharp Masking → YOLOv8n"
)

mode = st.radio(
    "Mode",
    ["Demo Mode", "Experiment Mode"],
    horizontal=True,
)

clip_limit = st.slider(
    "CLAHE clip limit",
    min_value=1.0,
    max_value=3.0,
    value=1.2,
    step=0.1,
)

tile_size = st.select_slider(
    "CLAHE tile grid size",
    options=[4, 8, 16],
    value=8,
)

sharpen_amount = st.slider(
    "Sharpening amount",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.1,
)

blur_kernel = st.select_slider(
    "Unsharp blur kernel",
    options=[3, 5, 7],
    value=5,
)

detail_threshold = st.slider(
    "Detail threshold",
    min_value=0,
    max_value=30,
    value=5,
    step=1,
)


def jiading_preprocess(image):
    return mild_clahe_unsharp(
        image,
        clip_limit=clip_limit,
        tile_size=tile_size,
        sharpen_amount=sharpen_amount,
        blur_kernel=blur_kernel,
        detail_threshold=detail_threshold,
    )


if mode == "Experiment Mode":

    st.info(
        "Mild LAB-CLAHE and threshold-controlled unsharp "
        "masking are applied to all training, validation and "
        "testing images. Image dimensions and object positions "
        "are preserved, so the original YOLO labels remain valid."
    )

    if st.button("Run small Jia Ding smoke test"):

        run_training_with_progress(
            SMOKE_EXPERIMENT_NAME,
            lambda progress: prepare_smoke_dataset(
                SMOKE_EXPERIMENT_NAME,
                jiading_preprocess,
                progress,
            ),
            SMOKE_SETTINGS["epochs"],
            "Jia Ding smoke test",
        )

    if st.button(
        "Prepare dataset and train Jia Ding (100 epochs)"
    ):

        run_training_with_progress(
            EXPERIMENT_NAME,
            lambda progress: prepare_processed_dataset(
                EXPERIMENT_NAME,
                jiading_preprocess,
                progress,
            ),
            100,
            "Jia Ding 100-epoch experiment",
        )

    render_training_output(
        SMOKE_EXPERIMENT_NAME
    )

    render_training_output(
        EXPERIMENT_NAME
    )


else:

    st.info(
        f"Detection model: results/{EXPERIMENT_NAME}/weights/best.pt"
    )

    image_file = st.file_uploader(
        "Upload PCB image",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if image_file is not None:

        try:
            original = uploaded_to_bgr(image_file)
            processed = jiading_preprocess(original)

            st.image(
                [original, processed],
                channels="BGR",
                caption=[
                    "Original PCB image",
                    "Mild LAB-CLAHE + Unsharp Masking",
                ],
            )

            if st.button("Run YOLOv8 Detection"):

                try:
                    result, objects, inference_time = detect(
                        processed,
                        EXPERIMENT_NAME,
                    )

                    st.image(
                        result,
                        channels="BGR",
                        caption="YOLOv8 detection result",
                    )

                    st.write(
                        f"Detected defects: {len(objects)}"
                    )

                    if inference_time is not None:
                        st.write(
                            f"Inference time: "
                            f"{inference_time:.2f} ms"
                        )

                    if objects:
                        st.subheader("Detected objects")

                        for class_name, confidence in objects:
                            st.write(
                                f"{class_name}: "
                                f"{confidence:.3f}"
                            )

                except Exception as error:
                    st.warning(
                        f"YOLO detection failed: {error}"
                    )

        except ValueError as error:
            st.warning(
                f"Unable to process image: {error}"
            )