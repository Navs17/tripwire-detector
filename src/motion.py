"""
Background subtraction module.
Separates moving foreground pixels from the static background.
"""
import cv2


def create_background_subtractor(history=500, var_threshold=16, detect_shadows=True):
    """
    Create a MOG2 background subtractor.
    
    MOG2 (Mixture of Gaussians v2) models each pixel as a mixture of Gaussian
    distributions. Pixels that don't match the learned model are flagged as
    foreground. It's adaptive — slow lighting changes get absorbed into the
    background over time.
    
    Args:
        history: number of frames used to build the background model
        var_threshold: sensitivity. Lower = more sensitive to motion
        detect_shadows: if True, shadows are marked in gray (value 127) instead of white
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )


def clean_mask(mask, kernel_size=5):
    """
    Clean up a foreground mask using morphological operations.
    
    Background subtraction tends to produce salt-and-pepper noise and small
    holes inside moving objects. Opening removes noise; closing fills holes.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Remove shadows (gray pixels) — treat only solid white as foreground
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    # Opening: erode then dilate — removes specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Closing: dilate then erode — fills small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask