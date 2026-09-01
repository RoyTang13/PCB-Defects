from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET
from config import CLASS_NAMES, DATASET_DIR

def xml_to_yolo(xml_path, label_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size"); width, height = float(size.findtext("width")), float(size.findtext("height"))
    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_NAMES: continue
        box = obj.find("bndbox"); xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
        xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
        x, y, w, h = ((xmin+xmax)/2/width, (ymin+ymax)/2/height, (xmax-xmin)/width, (ymax-ymin)/height)
        lines.append(f"{CLASS_NAMES.index(name)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    Path(label_path).parent.mkdir(parents=True, exist_ok=True)
    Path(label_path).write_text("\n".join(lines), encoding="utf-8")

def convert_and_split(source_images, source_xml, destination=DATASET_DIR, seed=42):
    """Create one reproducible 70/15/15 split. Run once and reuse it for all experiments."""
    images = [p for p in Path(source_images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    random.Random(seed).shuffle(images); n = len(images); cuts = (int(n*.70), int(n*.85))
    for split, group in zip(("train", "val", "test"), (images[:cuts[0]], images[cuts[0]:cuts[1]], images[cuts[1:]])):
        for image in group:
            target_img = Path(destination) / "images" / split / image.name
            target_lbl = Path(destination) / "labels" / split / f"{image.stem}.txt"
            target_img.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(image, target_img)
            xml = Path(source_xml) / f"{image.stem}.xml"
            if xml.exists(): xml_to_yolo(xml, target_lbl)
