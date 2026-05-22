"""
Contour-based object detection from a binary motion mask.
"""
import cv2


def find_objects(mask, min_area=500):
    """
    Find moving objects in a binary mask.
    
    Args:
        mask: single-channel binary image (foreground = 255, background = 0)
        min_area: contours smaller than this (in pixels) are discarded as noise
    
    Returns:
        List of dicts, each containing:
            'bbox': (x, y, w, h) bounding box
            'centroid': (cx, cy) center point
            'area': contour area in pixels
            'contour': the raw contour points
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2
        cy = y + h // 2
        
        objects.append({
            'bbox': (x, y, w, h),
            'centroid': (cx, cy),
            'area': area,
            'contour': c,
        })
    
    return objects


def draw_objects(frame, objects):
    """
    Draw bounding boxes and centroids on a frame.
    Returns the annotated frame (modifies in place too).
    """
    for obj in objects:
        x, y, w, h = obj['bbox']
        cx, cy = obj['centroid']
        
        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Centroid
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # Area label
        cv2.putText(frame, f"A={int(obj['area'])}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame