from pathlib import Path
import shutil
import random
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from config import ROOT, RESULTS_DIR, MODELS_DIR, YOLO_SETTINGS, SMOKE_SETTINGS, DATASET_DIR, PROCESSED_DIR, CLASS_NAMES

@st.cache_resource(show_spinner=False)
def load_model(weights):
    return YOLO(weights)

def trained_weights(experiment="baseline"):
    """Prefer full weights, then matching smoke-test weights, then a manual fallback."""
    smoke_name = "smoke_balanced" if experiment == "baseline" else f"{experiment}_smoke_balanced"
    candidates = (RESULTS_DIR / experiment / "weights" / "best.pt", RESULTS_DIR / smoke_name / "weights" / "best.pt", MODELS_DIR / "best.pt")
    return next((path for path in candidates if path.is_file()), None)

def detect(image, experiment="baseline", weights=None):
    selected = Path(weights) if weights else trained_weights(experiment)
    if selected is None:
        raise FileNotFoundError(
            f"No trained PCB model found for '{experiment}'. Train this experiment first; expected "
            f"{RESULTS_DIR / experiment / 'weights' / 'best.pt'}."
        )
    model = load_model(str(selected))
    result = model.predict(image, conf=YOLO_SETTINGS["conf"], iou=YOLO_SETTINGS["iou"], verbose=False)[0]
    return result.plot(), [(result.names[int(box.cls[0])], float(box.conf[0])) for box in result.boxes], result.speed.get("inference")

def train_experiment(experiment, data_yaml=None, epochs=None):
    model = YOLO("yolov8n.pt")
    return model.train(data=str(data_yaml or ROOT / "data.yaml"), project=str(RESULTS_DIR), name=experiment,
                       exist_ok=True, epochs=epochs or YOLO_SETTINGS["epochs"], imgsz=YOLO_SETTINGS["imgsz"],
                       batch=YOLO_SETTINGS["batch"], seed=YOLO_SETTINGS["seed"])

def run_training_with_progress(experiment, prepare_dataset, epochs, label):
    """Show Streamlit progress while a dataset is prepared, then train YOLO."""
    status = st.status(f"Starting {label}", expanded=True)
    progress = st.progress(0, text="Preparing dataset...")
    try:
        def update(done, total, message):
            progress.progress(done / total, text=f"{message} ({done}/{total})")
        data_yaml = prepare_dataset(update)
        progress.progress(1.0, text="Preprocessing complete.")
        status.write(f"Dataset ready. Training YOLOv8n for {epochs} epochs.")
        train_experiment(experiment, data_yaml, epochs=epochs)
        status.update(label=f"{label} completed", state="complete", expanded=False)
    except Exception as error:
        status.update(label=f"{label} failed", state="error", expanded=True)
        st.error(str(error))

def balanced_smoke_images():
    """Return deterministic, class-balanced source images for each fixed split."""
    selected = {}
    for split in ("train", "val", "test"):
        count = SMOKE_SETTINGS[f"{split}_per_class"]
        grouped = {class_id: [] for class_id in range(len(CLASS_NAMES))}
        for image in sorted((DATASET_DIR / "images" / split).glob("*")):
            label = DATASET_DIR / "labels" / split / f"{image.stem}.txt"
            if not label.exists(): continue
            class_ids = {int(line.split()[0]) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()}
            # Dataset images are defect-class-specific. Assign a deterministic primary class if needed.
            if class_ids: grouped[min(class_ids)].append(image)
        image_paths = []
        for class_id, choices in grouped.items():
            if len(choices) < count:
                raise ValueError(f"{split} has only {len(choices)} images for {CLASS_NAMES[class_id]}; needs {count}.")
            image_paths.extend(random.Random(YOLO_SETTINGS["seed"] + class_id).sample(choices, count))
        selected[split] = image_paths
    return selected

def write_dataset_yaml(output):
    yaml_path = output / "data.yaml"; names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n", encoding="utf-8")
    return yaml_path

def prepare_smoke_dataset(experiment="smoke_balanced", processor=None, on_progress=None):
    """Create a deterministic balanced subset, retaining every class in every split."""
    output = PROCESSED_DIR / experiment
    if output.exists(): shutil.rmtree(output)
    selected = balanced_smoke_images()
    total = sum(len(images) for images in selected.values())
    completed = 0
    for split, image_paths in selected.items():
        for image in image_paths:
            target_image = output / "images" / split / image.name; target_image.parent.mkdir(parents=True, exist_ok=True)
            if processor is None:
                shutil.copy2(image, target_image)
            else:
                original = cv2.imread(str(image))
                if original is None:
                    raise ValueError(f"Unreadable image: {image}")
                processed = processor(original)
                processed = processed[-1] if isinstance(processed, tuple) else processed
                if processed.ndim == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(target_image), processed)
            label = DATASET_DIR / "labels" / split / f"{image.stem}.txt"
            target_label = output / "labels" / split / label.name; target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label, target_label)
            completed += 1
            if on_progress:
                on_progress(completed, total, f"Preprocessing {split}: {image.name}")
    return write_dataset_yaml(output)

def prepare_processed_dataset(experiment, processor, on_progress=None):
    """Build a geometry-preserving copy of the fixed split and return its YAML file."""
    output = PROCESSED_DIR / experiment
    images_in, labels_in = DATASET_DIR / "images", DATASET_DIR / "labels"
    if not images_in.exists() or not any(images_in.rglob("*")):
        raise FileNotFoundError("Dataset images are missing. Convert/split the dataset before running an experiment.")
    if output.exists(): shutil.rmtree(output)
    shutil.copytree(labels_in, output / "labels")
    from utils.preprocessing import process_directory
    process_directory(images_in, output / "images", processor, on_progress)
    yaml_path = output / "data.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n", encoding="utf-8")
    return yaml_path

def prepare_manas_dataset(reference_root=None, kernel_size=5, iterations=1, on_progress=None):
    """Create Manas data from same-named normal reference images in reference_images/<split>."""
    reference_root = Path(reference_root or DATASET_DIR / "reference_images")
    if not reference_root.exists():
        raise FileNotFoundError("Create dataset/reference_images/train, val and test with same-named normal PCB references.")
    output = PROCESSED_DIR / "manas"
    if output.exists(): shutil.rmtree(output)
    shutil.copytree(DATASET_DIR / "labels", output / "labels")
    from utils.preprocessing import subtraction_morphology, IMAGE_EXTENSIONS
    paths = [(split, path) for split in ("train", "val", "test") for path in (DATASET_DIR / "images" / split).glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    for completed, (split, defective_path) in enumerate(paths, start=1):
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
            if on_progress: on_progress(completed, len(paths), f"Processing {split}: {defective_path.name}")
    yaml_path = output / "data.yaml"; names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n", encoding="utf-8")
    return yaml_path

def prepare_manas_smoke_dataset(reference_root=None, kernel_size=5, iterations=1, on_progress=None):
    """Build a class-balanced Manas smoke dataset from same-named normal references."""
    reference_root = Path(reference_root or DATASET_DIR / "reference_images")
    if not reference_root.exists():
        raise FileNotFoundError("Create dataset/reference_images/train, val and test with same-named normal PCB references.")
    output = PROCESSED_DIR / "manas_smoke_balanced"
    if output.exists(): shutil.rmtree(output)
    from utils.preprocessing import subtraction_morphology
    selected = balanced_smoke_images(); total = sum(len(paths) for paths in selected.values()); completed = 0
    for split, defective_paths in selected.items():
        for defective_path in defective_paths:
            reference_path = reference_root / split / defective_path.name
            if not reference_path.exists():
                raise FileNotFoundError(f"Missing paired reference: {reference_path}")
            reference, defective = cv2.imread(str(reference_path)), cv2.imread(str(defective_path))
            if reference is None or defective is None:
                raise ValueError(f"Unreadable image pair: {defective_path.name}")
            _, _, _, morphology, _, _ = subtraction_morphology(reference, defective, kernel_size, iterations)
            destination = output / "images" / split / defective_path.name; destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), cv2.cvtColor(morphology, cv2.COLOR_GRAY2BGR))
            source_label = DATASET_DIR / "labels" / split / f"{defective_path.stem}.txt"
            target_label = output / "labels" / split / source_label.name; target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_label, target_label)
            completed += 1
            if on_progress: on_progress(completed, total, f"Processing {split}: {defective_path.name}")
    return write_dataset_yaml(output)

def uploaded_to_bgr(uploaded):
    raw = uploaded.getvalue(); image = cv2.imdecode(__import__("numpy").frombuffer(raw, __import__("numpy").uint8), cv2.IMREAD_COLOR)
    if image is None: raise ValueError("The uploaded file is not a valid image.")
    return image

def render_training_output(experiment, processed_dataset=None):
    """Render genuine Ultralytics metrics and visual artifacts for a completed run."""
    run_dir = RESULTS_DIR / experiment
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return
    st.divider()
    st.subheader(f"Training output — {experiment}")
    frame = pd.read_csv(csv_path)
    final = frame.iloc[-1]
    metric_columns = {
        "Precision": "metrics/precision(B)", "Recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)", "mAP50-95": "metrics/mAP50-95(B)",
    }
    metrics = st.columns(4)
    for column, (label, key) in zip(metrics, metric_columns.items()):
        value = final.get(key)
        column.metric(label, "N/A" if pd.isna(value) else f"{value:.3f}")
    st.caption("These are the final validation metrics generated by YOLO; smoke-test metrics are only for checking the workflow.")

    processed_root = Path(processed_dataset) if processed_dataset else PROCESSED_DIR / experiment
    sample = next((processed_root / "images" / "train").glob("*"), None) if (processed_root / "images" / "train").exists() else None
    if sample:
        original = DATASET_DIR / "images" / "train" / sample.name
        image_cols = st.columns(2)
        if original.exists(): image_cols[0].image(str(original), caption="Original training image")
        image_cols[1].image(str(sample), caption="Preprocessed training image")

    artifacts = [("Training curves", run_dir / "results.png"), ("Confusion matrix", run_dir / "confusion_matrix.png"),
                 ("Validation labels", run_dir / "val_batch0_labels.jpg"), ("Validation predictions", run_dir / "val_batch0_pred.jpg")]
    for left, right in zip(artifacts[::2], artifacts[1::2]):
        columns = st.columns(2)
        for column, (caption, path) in zip(columns, (left, right)):
            if path.exists(): column.image(str(path), caption=caption)
