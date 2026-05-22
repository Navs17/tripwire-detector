import cv2
import time
import os

OUTPUT_PATH = "data/raw/test_walk.mp4"

# Make sure the folder exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError(
        f"VideoWriter failed to open {OUTPUT_PATH}. "
        f"Frame size: {width}x{height}, fps: {fps}."
    )

duration_seconds = 20
total_frames = duration_seconds * fps
print(f"Recording {duration_seconds} seconds at {width}x{height}...")
print("Press 'q' to stop early.")

start = time.time()
frame_count = 0
while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    elapsed = time.time() - start
    remaining = max(0, duration_seconds - elapsed)
    display = frame.copy()
    cv2.putText(display, f"REC {remaining:.1f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Recording...", display)
    frame_count += 1

    if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

if os.path.exists(OUTPUT_PATH):
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Saved: {OUTPUT_PATH} ({frame_count} frames, {size_mb:.2f} MB)")
else:
    print(f"ERROR: file was NOT written to {OUTPUT_PATH}")