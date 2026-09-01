# PCB Defect Detection and Image Processing Technique Comparison using YOLOv8

## Install and run

Create a virtual environment, then run `pip install -r requirements.txt`. Start the interface with `streamlit run app.py`.

## Dataset setup

Download the Kaggle PCB Defects dataset. If it has Pascal VOC XML labels, place source images and XML files in separate folders, then run `convert_and_split(source_images, source_xml)` from `utils.dataset_utils`. This creates the fixed, seed-42 70/15/15 split in `dataset/`; run it once and reuse that split for every experiment.

`data.yaml` points at this dataset and contains the six expected classes. The converter writes normalized YOLO `class x_center y_center width height` annotations.

## Training

In every page, choose **Experiment Mode** and run training. All experiments use YOLOv8n, 100 epochs, 640 image size, batch 16, and seed 42. Results are kept separately in `results/<experiment>/`; metrics and confusion matrices are read by the Comparison page.

For member datasets, use `utils.preprocessing.process_directory` with a geometry-preserving preprocessing function and copy the original labels unchanged. Do not create a new split. Manas requires correctly paired normal/reference images for valid full-dataset preprocessing.

For Manas, put same-named normal reference images in `dataset/reference_images/train`, `val`, and `test`; the page aligns each pair with ORB before subtraction. This explicit pairing avoids subtracting unrelated boards.

## Demo and comparison

Choose **Demo Mode** to inspect each pipeline and detections on an uploaded image. Use the **Comparison** page after training; it never invents metrics and displays N/A until Ultralytics has created a results file.
