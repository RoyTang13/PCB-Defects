# Compare all preprocessing pipelines on an uploaded PCB image.
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from utils.manas_preprocessing import subtraction_morphology
from utils.preprocessing import gaussian_colour, mild_clahe_unsharp, nlm_edge_contour
from utils.yolo_utils import detect, uploaded_to_bgr


def style_pdf_table(table, row_height, font_size):
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, column), cell in table.get_celld().items():
        cell.set_height(row_height)
        cell.PAD = 0.08
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold", ha="center")
        else:
            cell.set_facecolor("#f1f5f9" if row % 2 == 0 else "white")
            cell.set_text_props(color="#1f2937", ha="left" if column == 0 else "center")
        cell.set_edgecolor("#94a3b8")
        cell.set_linewidth(0.8)


st.title("All preprocessing comparison")
st.caption(
    "Apply every preprocessing pipeline to one PCB defect image, then compare the resulting YOLO detections."
)
st.info(
    "Manas subtraction + morphology requires a matching normal/reference PCB image. "
    "The other three pipelines run from the defect image alone."
)

with st.form("all_preprocessing_form"):
    # Collect the images needed by the comparison pipelines.
    defect_file = st.file_uploader(
        "Defect PCB image",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    reference_file = st.file_uploader(
        "Matching normal/reference PCB image (required for Manas)",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    run_all = st.form_submit_button("Run all preprocessing pipelines")

if run_all:
    if defect_file is None:
        st.error("Upload a defect PCB image first.")
    else:
        # Decode the uploaded defect image once for all pipelines.
        defective = uploaded_to_bgr(defect_file)
        st.subheader("Uploaded defect image")
        st.image(defective, channels="BGR", caption="Original defect PCB image")

        # Build the processed input for each independent pipeline.
        pipelines = [
            (
                "Gaussian filtering + colour segmentation",
                "leyi",
                gaussian_colour(defective)[2],
                "Gaussian kernel 5 with the full HSV range used during training.",
            ),
            (
                "Non-local means + edge / contour",
                "natasha",
                nlm_edge_contour(defective)[2],
                "NLM denoising followed by Canny edges and contours.",
            ),
            (
                "Mild LAB-CLAHE + unsharp masking",
                "jiading_100epochs",
                mild_clahe_unsharp(defective),
                "Mild contrast enhancement with threshold-controlled sharpening.",
            ),
        ]

        if reference_file is not None:
            # Add Manas when a matching reference image is available.
            reference = uploaded_to_bgr(reference_file)
            _, _, _, morphology, aligned_ok, alignment_message = subtraction_morphology(reference, defective)
            highlighted = defective.copy()
            highlighted[morphology > 0] = (0, 0, 255)
            colour_input = cv2.addWeighted(defective, 0.80, highlighted, 0.20, 0)
            pipelines.append(
                (
                    "Subtraction + morphology",
                    "manas",
                    colour_input,
                    f"{alignment_message} Colour-preserved YOLO input.",
                )
            )
            if not aligned_ok:
                st.warning("Manas alignment used a fallback. Its result may be less reliable.")
        else:
            st.warning("Manas was skipped because no matching normal/reference image was uploaded.")

        comparison_results = []
        # Run detection on every processed image and collect its results.
        for title, experiment, processed, detail in pipelines:
            with st.container(border=True):
                st.subheader(title)
                st.caption(detail)
                processed_column, detection_column = st.columns(2)
                processed_column.image(processed, channels="BGR", caption="After preprocessing")
                try:
                    detected, objects, inference_time = detect(processed, experiment)
                    detection_column.image(detected, channels="BGR", caption="YOLOv8 detection result")
                    inference_label = "N/A" if inference_time is None else f"{inference_time:.1f} ms"
                    comparison_results.append(
                        {
                            "title": title,
                            "detail": detail,
                            "processed": processed,
                            "detected": detected,
                            "objects": objects,
                            "inference_label": inference_label,
                        }
                    )
                    metrics = detection_column.columns(2)
                    metrics[0].metric("Detected defects", len(objects))
                    metrics[1].metric("Inference", inference_label)
                    if objects:
                        detection_column.dataframe(
                            {
                                "Defect": [label for label, _ in objects],
                                "Confidence": [round(confidence, 3) for _, confidence in objects],
                            },
                            hide_index=True,
                        )
                    else:
                        detection_column.info("No defect reached the confidence threshold.")
                except Exception as error:
                    comparison_results.append(
                        {
                            "title": title,
                            "detail": detail,
                            "processed": processed,
                            "error": str(error),
                        }
                    )
                    detection_column.error(f"Detection could not run: {error}")

        # Create a PDF report containing the comparison results.
        pdf_buffer = BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            summary_figure = plt.figure(figsize=(11.69, 8.27))
            summary_figure.text(0.08, 0.90, "PCB preprocessing comparison", fontsize=24, weight="bold")
            summary_figure.text(0.08, 0.85, f"Pipelines evaluated: {len(comparison_results)}", fontsize=12)
            summary_rows = []
            for result in comparison_results:
                if "error" in result:
                    summary_rows.append([result["title"], "Detection failed", "N/A"])
                else:
                    summary_rows.append(
                        [result["title"], str(len(result["objects"])), result["inference_label"]]
                    )
            summary_table = summary_figure.add_axes([0.08, 0.46, 0.84, 0.28])
            summary_table.axis("off")
            summary_table_object = summary_table.table(
                cellText=summary_rows,
                colLabels=["Pipeline", "Detected defects", "Inference"],
                colWidths=[0.56, 0.22, 0.22],
                loc="center",
                cellLoc="left",
                bbox=[0, 0, 1, 1],
            )
            style_pdf_table(summary_table_object, row_height=0.16, font_size=12)
            pdf.savefig(summary_figure)
            plt.close(summary_figure)

            for result in comparison_results:
                figure = plt.figure(figsize=(11.69, 8.27))
                layout = figure.add_gridspec(2, 2, height_ratios=[0.64, 0.36], hspace=0.12)
                axes = [figure.add_subplot(layout[0, 0]), figure.add_subplot(layout[0, 1])]
                figure.suptitle(result["title"], fontsize=18, weight="bold", y=0.96)
                axes[0].imshow(cv2.cvtColor(result["processed"], cv2.COLOR_BGR2RGB))
                axes[0].set_title("After preprocessing")
                axes[0].axis("off")
                if "error" in result:
                    axes[1].text(0.5, 0.5, f"Detection failed:\n{result['error']}", ha="center", va="center", wrap=True)
                    detail_rows = [["Detection failed", result["error"]]]
                else:
                    axes[1].imshow(cv2.cvtColor(result["detected"], cv2.COLOR_BGR2RGB))
                    axes[1].set_title("YOLOv8 detection result")
                    detail_rows = [
                        [label, f"{confidence:.3f}"]
                        for label, confidence in result["objects"]
                    ] or [["No defect", "-"]]
                axes[1].axis("off")
                detail_table = figure.add_subplot(layout[1, :])
                detail_table.axis("off")
                detail_table_object = detail_table.table(
                    cellText=detail_rows,
                    colLabels=["Defect", "Confidence"],
                    colWidths=[0.68, 0.32],
                    loc="center",
                    cellLoc="left",
                )
                style_pdf_table(detail_table_object, row_height=0.14, font_size=10)
                figure.text(0.08, 0.02, result["detail"], fontsize=8, wrap=True)
                pdf.savefig(figure)
                plt.close(figure)

        st.download_button(
            "Export comparison as PDF",
            data=pdf_buffer.getvalue(),
            file_name="pcb_preprocessing_comparison.pdf",
            mime="application/pdf",
        )
