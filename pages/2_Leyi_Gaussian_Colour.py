# Gaussian filtering and HSV colour segmentation experiment.
import streamlit as st
from utils.yolo_utils import uploaded_to_bgr, detect, prepare_processed_dataset, render_training_output, run_training_with_progress
from utils.preprocessing import gaussian_colour
st.title("Leyi — Gaussian Filtering + Colour Segmentation")
st.caption("Original → Gaussian noise reduction → HSV colour segmentation → YOLOv8n")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
if mode == "Experiment Mode":
    # Prepare the processed dataset and train the Leyi model.
    st.info("The fixed split is copied and processed without geometry changes, so its YOLO labels are reused unchanged.")
    if st.button("Prepare dataset and train Leyi (100 epochs)"):
        run_training_with_progress("leyi", lambda progress: prepare_processed_dataset("leyi", lambda x: gaussian_colour(x)[2], progress), 100, "Leyi full experiment")
    render_training_output("leyi")
else:
    # Set Gaussian and HSV parameters for the demo image.
    k = st.select_slider("Gaussian kernel", [3, 5, 7], value=5); cols = st.columns(2)
    lo = tuple(cols[0].slider(x, 0, 179 if x == "H minimum" else 255, 0) for x in ("H minimum", "S minimum", "V minimum"))
    hi = tuple(cols[1].slider(x, 0, 179 if x == "H maximum" else 255, 179 if x == "H maximum" else 255) for x in ("H maximum", "S maximum", "V maximum"))
    inspect_low_confidence = st.toggle(
        "Inspect low-confidence candidates",
        value=False,
        help="Use this only to diagnose missed detections; low-confidence boxes are not reliable results.",
    )
    confidence = 0.25
    if inspect_low_confidence:
        confidence = st.slider(
            "Diagnostic confidence threshold",
            min_value=0.01,
            max_value=0.25,
            value=0.25,
            step=0.01,
        )
    st.caption("The trained Leyi model uses Gaussian kernel 5 and the full HSV range. Keep these defaults for the closest match to training.")
    file = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png", "bmp"])
    if file:
        # Apply Gaussian filtering and HSV colour segmentation.
        image = uploaded_to_bgr(file); blurred, mask, segmented = gaussian_colour(image, k, lo, hi)
        for col, pic, label in zip(st.columns(4), (image, blurred, mask, segmented), ("Original", "Gaussian filtered", "HSV mask", "Segmented")): col.image(pic, channels="BGR" if pic.ndim == 3 else "GRAY", caption=label)
        if st.button("Run YOLOv8 Detection"):
            # Detect defects in the segmented image.
            try:
                result, objects, _ = detect(segmented, "leyi", confidence=confidence)
                st.image(result, channels="BGR", caption=f"Leyi detections at confidence ≥ {confidence:.2f}")
                st.write(f"Defects: {len(objects)}")
                if objects:
                    st.dataframe({"Defect": [name for name, _ in objects], "Confidence": [round(score, 3) for _, score in objects]}, hide_index=True)
                else:
                    st.warning("No reliable detection reached the confidence threshold. You can enable diagnostic mode to inspect weak candidates, but they should not be treated as confirmed defects.")
            except Exception as e: st.warning(str(e))
