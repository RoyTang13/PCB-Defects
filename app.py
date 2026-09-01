import streamlit as st
from config import YOLO_SETTINGS

st.set_page_config(page_title="PCB Defect Detection", page_icon="🔬", layout="wide")
st.title("PCB Defect Detection and Preprocessing Comparison")
st.markdown("Compare **YOLOv8n** under one fixed protocol; only preprocessing differs between experiments.")
st.info(f"Fixed settings — epochs: {YOLO_SETTINGS['epochs']}, image size: {YOLO_SETTINGS['imgsz']}, batch: {YOLO_SETTINGS['batch']}, seed: {YOLO_SETTINGS['seed']}, confidence: {YOLO_SETTINGS['conf']}, IoU: {YOLO_SETTINGS['iou']}.")
st.markdown("Use the sidebar to open the baseline, team-member pipelines, or final comparison.")
