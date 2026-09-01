from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "processed_datasets"
CLASS_NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
EXPERIMENTS = {
    "baseline": "Baseline YOLOv8", "leyi": "Gaussian + Colour", "natasha": "NLM + Edge/Contour",
    "jiading": "CLAHE + Template", "manas": "Subtraction + Morphology",
}
YOLO_SETTINGS = {"epochs": 100, "imgsz": 640, "batch": 16, "seed": 42, "conf": 0.25, "iou": 0.50}

for folder in (DATASET_DIR, RESULTS_DIR, MODELS_DIR, PROCESSED_DIR):
    folder.mkdir(parents=True, exist_ok=True)
