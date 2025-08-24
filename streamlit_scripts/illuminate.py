# luminosity_utils.py

import cv2
import numpy as np

def compute_luminosity(image_bgr):
    """
    Compute the average luminosity of a given image.

    Args:
        image_bgr (np.ndarray): Cropped image in BGR format.

    Returns:
        float: Mean grayscale brightness value (0–255).
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("Invalid input image. Must be a numpy ndarray.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def compare_luminosity(off_boxes, on_boxes, threshold=0.01):
    """
    Compare brightness of brake light crops between off/on states.

    Args:
        off_boxes (List[np.ndarray]): Cropped brake lights when brakes are OFF.
        on_boxes (List[np.ndarray]): Cropped brake lights when brakes are ON.
        threshold (float): Brightness delta threshold to determine functionality.

    Returns:
        List[Tuple[int, float, float, float, str]]:
            Index, luminosity_off, luminosity_on, delta, status message
    """
    if not off_boxes or not on_boxes:
        raise ValueError("Both off_boxes and on_boxes must be non-empty lists of cropped images.")

    results = []
    min_length = min(len(off_boxes), len(on_boxes))

    for i in range(min_length):
        lum_off = compute_luminosity(off_boxes[i])
        lum_on = compute_luminosity(on_boxes[i])
        delta = lum_on - lum_off
        status = "✅ Functional" if delta >= threshold else "❌ Non-functional"

        results.append((i + 1, lum_off, lum_on, delta, status))

    return results
