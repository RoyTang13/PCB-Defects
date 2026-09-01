from pathlib import Path
import shutil
import cv2
import streamlit as st
from ultralytics import YOLO
from config import ROOT, RESULTS_DIR, YOLO_SETTINGS, DATASET_DIR, PROCESSED_DIR, CLASS_NAMES

@st.cache_resource(show_spinner=False)
def load_model(weights="yolov8n.pt"):
    return YOLO(weights)

def detect(image, weights="yolov8n.pt"):
    model = load_model(weights)
    result = model.predict(image, conf=YOLO_SETTINGS["conf"], iou=YOLO_SETTINGS["iou"], verbose=False)[0]
    return result.plot(), [(result.names[int(box.cls[0])], float(box.conf[0])) for box in result.boxes], result.speed.get("inference")

def train_experiment(experiment, data_yaml=None):
    model = YOLO("yolov8n.pt")
    return model.train(data=str(data_yaml or ROOT / "data.yaml"), project=str(RESULTS_DIR), name=experiment,
                       exist_ok=True, **{k: v for k, v in YOLO_SETTINGS.items() if k in {"epochs", "imgsz", "batch", "seed"}})

def prepare_processed_dataset(experiment, processor):
    """Build a geometry-preserving copy of the fixed split and return its YAML file."""
    output = PROCESSED_DIR / experiment
    images_in, labels_in = DATASET_DIR / "images", DATASET_DIR / "labels"
    if not images_in.exists() or not any(images_in.rglob("*")):
        raise FileNotFoundError("Dataset images are missing. Convert/split the dataset before running an experiment.")
    if output.exists(): shutil.rmtree(output)
    shutil.copytree(labels_in, output / "labels")
    from utils.preprocessing import process_directory
    process_directory(images_in, output / "images", processor)
    yaml_path = output / "data.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n", encoding="utf-8")
    return yaml_path

def prepare_manas_dataset(reference_root=None, kernel_size=5, iterations=1):
    """Create Manas data from same-named normal reference images in reference_images/<split>."""
    reference_root = Path(reference_root or DATASET_DIR / "reference_images")
    if not reference_root.exists():
        raise FileNotFoundError("Create dataset/reference_images/train, val and test with same-named normal PCB references.")
    output = PROCESSED_DIR / "manas"
    if output.exists(): shutil.rmtree(output)
    shutil.copytree(DATASET_DIR / "labels", output / "labels")
    from utils.preprocessing import subtraction_morphology, IMAGE_EXTENSIONS
    for split in ("train", "val", "test"):
        for defective_path in (DATASET_DIR / "images" / split).glob("*"):
            if defective_path.suffix.lower() not in IMAGE_EXTENSIONS: continue
            ref_path = reference_root / split / defective_path.name
            if not ref_path.exists(): raise FileNotFoundError(f"Missing paired reference: {ref_path}")
            ref, defective = cv2.imread(str(ref_path)), cv2.imread(str(defective_path))
            if ref is None or defective is None: raise ValueError(f"Unreadable image pair: {defective_path.name}")
            _, _, _, morph, _, _ = subtraction_morphology(ref, defective, kernel_size, iterations)
            # YOLO requires three-channel files; morphology is duplicated without changing geometry.
            processed = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
            destination = output / "images" / split / defective_path.name; destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), processed)
    yaml_path = output / "data.yaml"; names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n", encoding="utf-8")
    return yaml_path

def uploaded_to_bgr(uploaded):
    raw = uploaded.getvalue(); image = cv2.imdecode(__import__("numpy").frombuffer(raw, __import__("numpy").uint8), cv2.IMREAD_COLOR)
    if image is None: raise ValueError("The uploaded file is not a valid image.")
    return image
