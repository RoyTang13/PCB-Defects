from ultralytics import YOLO

model = YOLO(
    "results/jiading_100epochs/weights/best.pt"
)

metrics = model.val(
    data="processed_datasets/jiading_100epochs/data.yaml"
)

print("\n===== BEST.PT VALIDATION RESULT =====")
print(f"Precision: {metrics.box.mp:.5f}")
print(f"Recall: {metrics.box.mr:.5f}")
print(f"mAP50: {metrics.box.map50:.5f}")
print(f"mAP50-95: {metrics.box.map:.5f}")