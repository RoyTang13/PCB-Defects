"""Classical reference-subtraction preprocessing used only by the Manas module."""

import cv2
import numpy as np


def align_reference(reference, target):
    """Align the normal reference image to its defective counterpart with ORB."""
    orb = cv2.ORB_create(3000)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_ref, None)
    keypoints_target, descriptors_target = orb.detectAndCompute(gray_target, None)
    fallback = cv2.resize(reference, (target.shape[1], target.shape[0]))
    if descriptors_ref is None or descriptors_target is None:
        return fallback, False, "No ORB descriptors found; resized fallback used."
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(descriptors_ref, descriptors_target)
    if len(matches) < 4:
        return fallback, False, "Too few ORB matches; resized fallback used."
    matches = sorted(matches, key=lambda match: match.distance)[:100]
    source = np.float32([keypoints_ref[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    destination = np.float32([keypoints_target[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    if homography is None or mask is None or int(mask.sum()) < 4:
        return fallback, False, "Homography failed; resized fallback used."
    return cv2.warpPerspective(reference, homography, (target.shape[1], target.shape[0])), True, "ORB homography alignment applied."


def subtraction_morphology(reference, defective, kernel_size=5, iterations=1):
    """Align, subtract, threshold, then clean a reference/defect image pair."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Morphology kernel size must be an odd number of at least 3.")
    if iterations < 1:
        raise ValueError("Morphology iterations must be at least 1.")
    aligned, success, message = align_reference(reference, defective)
    difference = cv2.absdiff(aligned, defective)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    morphology = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    morphology = cv2.morphologyEx(morphology, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return aligned, difference, binary, morphology, success, message


def find_candidate_regions(mask, minimum_area=20):
    """Return bounding boxes for meaningful connected regions in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= minimum_area:
            x, y, width, height = cv2.boundingRect(contour)
            regions.append({"x": x, "y": y, "width": width, "height": height, "area": round(float(area), 1)})
    return sorted(regions, key=lambda region: region["area"], reverse=True)
