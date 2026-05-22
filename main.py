"""
tripwire detector.
- Direction filtering: only alert on selected crossing direction
- Velocity gating: ignore too-slow (noise) and too-fast (glitch) tracks
- Persistence: track must exist for N frames before it can alert
- Per-track cooldown: prevent the same track from spamming alerts
"""
import math
import cv2
import numpy as np
from src.motion import create_background_subtractor, clean_mask
from src.detection import find_objects
from src.tracker import CentroidTracker
from src.tripwire import Tripwire, define_tripwire_interactively

VIDEO_PATH = "data/raw/test_walk.mp4"

# Detection params
MIN_OBJECT_AREA = 500

# False-positive filter params
MIN_TRACK_AGE = 5          # track must have existed >= 5 frames to alert
MIN_SPEED = 1.5            # pixels/frame — below this is treated as noise
MAX_SPEED = 100.0          # pixels/frame — above this is a tracking glitch
ALERT_COOLDOWN_FRAMES = 30 # same track cannot re-alert within this window
ALLOWED_DIRECTION = None   # set to +1 or -1 to restrict; None = allow both

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")

pt_a, pt_b = define_tripwire_interactively(first_frame)
tripwire = Tripwire(pt_a, pt_b, name="Perimeter")
print(f"Tripwire: {pt_a} -> {pt_b}")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
bg_subtractor = create_background_subtractor()
tracker = CentroidTracker(max_distance=80)

alert_count = 0
alert_flash_frames = 0
last_alert_frame = {}  # track_id -> frame number of last alert
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame_idx += 1
    
    raw_mask = bg_subtractor.apply(frame)
    clean = clean_mask(raw_mask)
    objects = find_objects(clean, min_area=MIN_OBJECT_AREA)
    tracks = tracker.update(objects)
    
    annotated = frame.copy()
    tripwire.draw(annotated)
    
    for tid, track in tracks.items():
        det = track['detection']
        x, y, w, h = det['bbox']
        cx, cy = track['centroid']
        vx, vy = track['velocity']
        speed = math.hypot(vx, vy)
        age = track['age']
        
        # Color box by whether track is "mature" enough to alert
        mature = age >= MIN_TRACK_AGE
        box_color = (0, 255, 0) if mature else (128, 128, 128)
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(annotated, f"ID {tid} age={age} v={speed:.1f}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # --- Crossing check with all three filters ---
        prev = track.get('prev_centroid')
        curr = track['centroid']
        direction = tripwire.crossed(prev, curr)
        
        if direction is None:
            continue
        
        # Filter 1: persistence
        if age < MIN_TRACK_AGE:
            print(f"[suppressed] track {tid}: too new (age={age})")
            continue
        
        # Filter 2: velocity gating
        if speed < MIN_SPEED:
            print(f"[suppressed] track {tid}: too slow (v={speed:.2f})")
            continue
        if speed > MAX_SPEED:
            print(f"[suppressed] track {tid}: implausibly fast (v={speed:.2f})")
            continue
        
        # Filter 3: direction
        if ALLOWED_DIRECTION is not None and direction != ALLOWED_DIRECTION:
            print(f"[suppressed] track {tid}: wrong direction ({direction})")
            continue
        
        # Filter 4: cooldown — don't double-alert the same track
        last = last_alert_frame.get(tid, -10**9)
        if frame_idx - last < ALERT_COOLDOWN_FRAMES:
            continue
        
        # All filters passed — fire alert
        alert_count += 1
        alert_flash_frames = 15
        last_alert_frame[tid] = frame_idx
        print(f"[ALERT #{alert_count}] track {tid} crossed at frame {frame_idx}, "
              f"dir={direction}, v={speed:.2f}, age={age}")
    
    if alert_flash_frames > 0:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (annotated.shape[1], annotated.shape[0]),
                      (0, 0, 255), 20)
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
        cv2.putText(annotated, "INTRUSION!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        alert_flash_frames -= 1
    
    cv2.putText(annotated, f"Alerts: {alert_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.imshow("Tripwire Detector v1.5 - press 'q' to quit", annotated)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal alerts: {alert_count}")