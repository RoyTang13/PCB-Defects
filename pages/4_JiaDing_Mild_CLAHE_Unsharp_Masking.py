# LAB contrast enhancement and unsharp masking experiment.
import cv2
import numpy as np
import streamlit as st

from utils.yolo_utils import (
    uploaded_to_bgr,
    detect,
    prepare_processed_dataset,
    render_training_output,
    run_training_with_progress,
)

from utils.preprocessing import mild_clahe_unsharp


EXPERIMENT_NAME = "jiading_100epochs"


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



def jiading_demo_stages(image):



    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel,
        )
    )

    clahe_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )


    blurred = cv2.GaussianBlur(
        clahe_image,
        (blur_kernel, blur_kernel),
        0,
    )

    detail = cv2.absdiff(
        clahe_image,
        blurred,
    )

    detail_gray = cv2.cvtColor(
        detail,
        cv2.COLOR_BGR2GRAY,
    )

    # Improve visibility for demonstration.
    # This only affects the displayed detail image.
    detail_display = cv2.normalize(
        detail_gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    # Stage 4: Threshold-controlled detail mask

    detail_mask = np.where(
        detail_gray >= detail_threshold,
        255,
        0,
    ).astype(np.uint8)

    # Stage 5: Actual final preprocessing

    final_image = mild_clahe_unsharp(
        image,
        clip_limit=clip_limit,
        tile_size=tile_size,
        sharpen_amount=sharpen_amount,
        blur_kernel=blur_kernel,
        detail_threshold=detail_threshold,
    )

    return (
        clahe_image,
        detail_display,
        detail_mask,
        final_image,
    )


# Experiment Mode

if mode == "Experiment Mode":

    st.info(
        "Mild LAB-CLAHE and threshold-controlled unsharp "
        "masking are applied to all training, validation and "
        "testing images. Image dimensions and object positions "
        "are preserved, so the original YOLO labels remain valid."
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
        EXPERIMENT_NAME
    )

else:

    st.info(
        f"Detection model: "
        f"results/{EXPERIMENT_NAME}/weights/best.pt"
    )

    image_file = st.file_uploader(
        "Upload PCB image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
        ],
    )

    if image_file is not None:

        try:

            original = uploaded_to_bgr(
                image_file
            )

            (
                clahe_image,
                detail_image,
                detail_mask,
                processed,
            ) = jiading_demo_stages(
                original
            )

            st.subheader("Preprocessing Stages")

            row1 = st.columns(3)

            with row1[0]:
                st.image(
                    original,
                    channels="BGR",
                    caption="Original",
                    use_container_width=True,
                )

            with row1[1]:
                st.image(
                    clahe_image,
                    channels="BGR",
                    caption="Mild LAB-CLAHE",
                    use_container_width=True,
                )

            with row1[2]:
                st.image(
                    detail_image,
                    caption="Extracted Detail",
                    use_container_width=True,
                )

            row2 = st.columns(2)

            with row2[0]:
                st.image(
                    detail_mask,
                    caption="Detail Mask",
                    use_container_width=True,
                )

            with row2[1]:
                st.image(
                    processed,
                    channels="BGR",
                    caption="Final Processed",
                    use_container_width=True,
                )

            if st.button(
                "Run YOLOv8 Detection"
            ):

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
                        f"Detected defects: "
                        f"{len(objects)}"
                    )

                    if inference_time is not None:

                        st.write(
                            f"Inference time: "
                            f"{inference_time:.2f} ms"
                        )

                    if objects:

                        st.subheader(
                            "Detected objects"
                        )

                        for class_name, confidence in objects:

                            st.write(
                                f"{class_name}: "
                                f"{confidence:.3f}"
                            )

                except Exception as error:

                    st.warning(
                        f"YOLO detection failed: "
                        f"{error}"
                    )

        except ValueError as error:

            st.warning(
                f"Unable to process image: "
                f"{error}"
            )