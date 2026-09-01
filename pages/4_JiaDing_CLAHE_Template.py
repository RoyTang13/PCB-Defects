import streamlit as st
from utils.yolo_utils import uploaded_to_bgr, detect, train_experiment, prepare_processed_dataset
from utils.preprocessing import clahe_lab, template_match
st.title("Jia Ding — CLAHE + Template Matching")
st.caption("Original → LAB CLAHE → template matching → YOLOv8n")
mode = st.radio("Mode", ["Demo Mode", "Experiment Mode"], horizontal=True)
if mode == "Experiment Mode":
    st.info("Template matching is a visual/diagnostic stage; dataset training uses the geometry-preserving CLAHE output.")
    if st.button("Prepare dataset and train Jia Ding (100 epochs)"):
        try: train_experiment("jiading", prepare_processed_dataset("jiading", lambda x: clahe_lab(x)))
        except Exception as e: st.error(str(e))
else:
    clip = st.slider("CLAHE clipLimit", 1.0, 10.0, 2.0); tile = st.select_slider("tileGridSize", [4, 8, 16], value=8)
    image_file = st.file_uploader("PCB target image", type=["jpg", "jpeg", "png", "bmp"]); template_file = st.file_uploader("Normal/reference template", type=["jpg", "jpeg", "png", "bmp"])
    if image_file and template_file:
        image, template = uploaded_to_bgr(image_file), uploaded_to_bgr(template_file); enhanced = clahe_lab(image, clip, tile)
        try:
            matched, score = template_match(enhanced, clahe_lab(template, clip, tile)); st.image([image, enhanced, matched], channels="BGR", caption=["Original", "CLAHE enhanced", f"Template result (score {score:.3f})"])
            if st.button("Run YOLOv8 Detection"):
                result, objects, _ = detect(enhanced, "jiading"); st.image(result, channels="BGR"); st.write(f"Defects: {len(objects)}")
        except ValueError as e: st.warning(str(e))
