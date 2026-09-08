from pathlib import Path
import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def gaussian_colour(image, kernel=5, lower=(0, 0, 0), upper=(179, 255, 255)):
    # Blur the image to reduce noise.
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
    # Convert to HSV and keep the selected colour range.
    mask = cv2.inRange(cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV), np.array(lower), np.array(upper))
    # Apply the colour mask to the blurred image.
    return blurred, mask, cv2.bitwise_and(blurred, blurred, mask=mask)

def nlm_edge_contour(image, threshold1=80, threshold2=160):
    # Remove noise while preserving image details.
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # Detect edges from the denoised grayscale image.
    edges = cv2.Canny(cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY), threshold1, threshold2)
    # Draw external contours on a copy of the denoised image.
    contour_image = denoised.copy()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    return denoised, edges, contour_image

def clahe_lab(image, clip_limit=2.0, tile_size=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    enhanced_l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)).apply(l)
    return cv2.cvtColor(cv2.merge((enhanced_l, a, b)), cv2.COLOR_LAB2BGR)

def clahe_lab(image, clip_limit=2.0, tile_size=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    enhanced_l = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    ).apply(l)

    return cv2.cvtColor(
        cv2.merge((enhanced_l, a, b)),
        cv2.COLOR_LAB2BGR,
    )


def mild_clahe_unsharp(
    image,
    clip_limit=1.2,
    tile_size=8,
    sharpen_amount=0.4,
    blur_kernel=5,
    detail_threshold=5,
):
    """
    Apply mild LAB-CLAHE followed by threshold-controlled
    unsharp masking without changing image geometry.
    """

    if blur_kernel < 3:
        blur_kernel = 3

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    # Convert the image to LAB colour space.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )

    # Enhance local contrast in the lightness channel.
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    # Convert the enhanced LAB image back to BGR.
    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

    # Create a blurred version for unsharp masking.
    blurred = cv2.GaussianBlur(
        enhanced_image,
        (blur_kernel, blur_kernel),
        0,
    )

    # Strengthen details using the blurred image.
    sharpened = cv2.addWeighted(
        enhanced_image,
        1.0 + sharpen_amount,
        blurred,
        -sharpen_amount,
        0,
    )

    # Build a mask for details above the threshold.
    detail_difference = cv2.absdiff(
        enhanced_image,
        blurred,
    )

    detail_mask = (
        cv2.cvtColor(
            detail_difference,
            cv2.COLOR_BGR2GRAY,
        )
        >= detail_threshold
    )

    output = enhanced_image.copy()
    output[detail_mask] = sharpened[detail_mask]

    return output

def mild_clahe_unsharp(
    image,
    clip_limit=1.2,
    tile_size=8,
    sharpen_amount=0.4,
    blur_kernel=5,
    detail_threshold=5,
):
    """
    Apply mild LAB-CLAHE followed by threshold-controlled
    unsharp masking without changing image geometry.
    """

    if blur_kernel < 3:
        blur_kernel = 3

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

    blurred = cv2.GaussianBlur(
        enhanced_image,
        (blur_kernel, blur_kernel),
        0,
    )

    sharpened = cv2.addWeighted(
        enhanced_image,
        1.0 + sharpen_amount,
        blurred,
        -sharpen_amount,
        0,
    )

    detail_difference = cv2.absdiff(
        enhanced_image,
        blurred,
    )

    detail_mask = (
        cv2.cvtColor(
            detail_difference,
            cv2.COLOR_BGR2GRAY,
        )
        >= detail_threshold
    )

    output = enhanced_image.copy()
    output[detail_mask] = sharpened[detail_mask]

    return output

def process_directory(source, destination, processor, on_progress=None):
    """Apply a geometry-preserving processor to every image, retaining directory layout."""
    source, destination = Path(source), Path(destination)
    paths = [path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    for index, path in enumerate(paths, start=1):
            image = cv2.imread(str(path))
            if image is not None:
                output = processor(image)
                output = output[-1] if isinstance(output, tuple) else output
                out_path = destination / path.relative_to(source)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), output)
                if on_progress: on_progress(index, len(paths), f"Preprocessing {path.name}")
