# Tripwire Detector
Real-time perimeter intrusion detection using computer vision techniques. Simulates a smart fence camera by detecting when objects cross a user-defined 
virtual tripwire line.

## Pipeline
- **MOG2 background subtraction** — adaptive Gaussian mixture model isolates moving foreground from a learned background.
- **Morphological cleanup** — opening and closing remove noise and fill holes in the motion mask.
- **Contour-based detection** — extracts moving objects with area-based filtering.
- **Centroid tracking** — greedy nearest-neighbor matching maintains object identity across frames.
- **Tripwire crossing** — signed cross-product test against a user-defined line segment, with directionality.
- **False-positive filtering** — track persistence, velocity gating, direction filter, and per-track cooldown.

## Setup
```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

## Usage
1. Record a test clip: `python record_clip.py`
2. Run the detector: `python main.py`
3. Click two points on the first frame to define the tripwire.
4. Watch as crossings trigger intrusion alerts.

## Tech stack
Python 3.11, OpenCV 4.10, NumPy.

## Roadmap
- [ ] Persistent alert logging (JSON + frame snapshots)
- [ ] Polygonal zones with loitering detection
- [ ] Quantitative evaluation (precision/recall on labeled clips)
- [ ] YOLO-based object classification baseline comparison
- [ ] Edge deployment (Jetson Nano / Raspberry Pi 5)
