from pathlib import Path
import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def gaussian_colour(image, kernel=5, lower=(0, 0, 0), upper=(179, 255, 255)):
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)
    mask = cv2.inRange(cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV), np.array(lower), np.array(upper))
    return blurred, mask, cv2.bitwise_and(blurred, blurred, mask=mask)

def nlm_edge_contour(image, threshold1=80, threshold2=160):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    edges = cv2.Canny(cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY), threshold1, threshold2)
    contour_image = denoised.copy()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    return denoised, edges, contour_image

def nlm_edge_contour_smoke(image, max_width=640):
    """Fast smoke-test variant: scale only before NLM; normalized YOLO labels remain valid."""
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
    return nlm_edge_contour(image)

def clahe_lab(image, clip_limit=2.0, tile_size=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    enhanced_l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)).apply(l)
    return cv2.cvtColor(cv2.merge((enhanced_l, a, b)), cv2.COLOR_LAB2BGR)

def template_match(target, template):
    if template.shape[0] > target.shape[0] or template.shape[1] > target.shape[1]:
        raise ValueError("Reference template must not be larger than the target image.")
    result = cv2.matchTemplate(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    visual = target.copy()
    h, w = template.shape[:2]
    cv2.rectangle(visual, location, (location[0] + w, location[1] + h), (0, 255, 0), 2)
    return visual, float(score)

def align_reference(reference, target):
    """Align reference to target via ORB/homography; returns fallback only if alignment fails."""
    orb = cv2.ORB_create(3000)
    gray_ref, gray_target = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(gray_ref, None); k2, d2 = orb.detectAndCompute(gray_target, None)
    if d1 is None or d2 is None: return cv2.resize(reference, (target.shape[1], target.shape[0])), False, "No ORB descriptors found; resized fallback used."
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 4: return cv2.resize(reference, (target.shape[1], target.shape[0])), False, "Too few ORB matches; resized fallback used."
    matches = sorted(matches, key=lambda x: x.distance)[:100]
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None or mask is None or int(mask.sum()) < 4: return cv2.resize(reference, (target.shape[1], target.shape[0])), False, "Homography failed; resized fallback used."
    return cv2.warpPerspective(reference, homography, (target.shape[1], target.shape[0])), True, "ORB homography alignment applied."

def subtraction_morphology(reference, defective, kernel_size=5, iterations=1):
    aligned, success, message = align_reference(reference, defective)
    difference = cv2.absdiff(aligned, defective)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return aligned, difference, binary, morph, success, message

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
