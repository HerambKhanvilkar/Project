import os
import tempfile
import requests
import cv2
from ultralytics import YOLO
from datetime import datetime, timezone

# ─── YOUR SUPABASE CONFIG ──────────────────────────────────────────────────────
SUPABASE_URL = "https://qnttrmrwrenlsnpwcrkl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFudHRybXJ3cmVubHNucHdjcmtsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzI1NTk4OCwiZXhwIjoyMDY4ODMxOTg4fQ.d20cXxyVbdmgO1F4Dvm4B2UTsJCWD37bReL9C-l1J0k"

# ─── LOAD MODEL ONCE ───────────────────────────────────────────────────────────
MODEL = YOLO("yolov8n.pt")

# ─── COCO CLASS IDS ────────────────────────────────────────────────────────────
PERSON_CLASS = 0
CAR_CLASS    = 2
FIRE_CLASS   = 43  # adjust if you have a custom fire/smoke class

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def download_file(url: str) -> str:
    """Stream-download URL to a temp file, return local path."""
    r = requests.get(url, stream=True, timeout=10)
    r.raise_for_status()
    ext = os.path.splitext(url)[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return path

def push_insight(insight: dict):
    """Insert one row into supabase insights via REST."""
    endpoint = f"{SUPABASE_URL}/rest/v1/insights"
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation"
    }
    r = requests.post(endpoint, headers=headers, json=insight, timeout=10)
    r.raise_for_status()
    return r.json()

# ─── CORE PREDICT (FIRST FRAME) ────────────────────────────────────────────────
def analyze_frame(frame, latitude, longitude):
    # resize & mild sharpen
    small = cv2.resize(frame, (320, 320))
    blur  = cv2.GaussianBlur(small, (0,0), sigmaX=2, sigmaY=2)
    inp   = cv2.addWeighted(small, 1.3, blur, -0.3, 0)

    # run YOLO
    results = MODEL(inp, conf=0.1, iou=0.45, augment=False)

    # counts
    pts = sum(int(b.cls)==PERSON_CLASS for r in results for b in r.boxes)
    cts = sum(int(b.cls)==CAR_CLASS    for r in results for b in r.boxes)
    fts = sum(int(b.cls)==FIRE_CLASS   for r in results for b in r.boxes)

    # classify
    if pts > 40:       dtype = "stampede"
    elif fts > 0:      dtype = "riot"
    elif cts > 0:      dtype = "accident"
    else:              dtype = "unknown"

    status = "SAFE" if pts <= 30 else "UNSAFE"

    insight = {
        "type":       dtype,
        "location":   None,
        "latitude":   latitude,
        "longitude":  longitude,
        "status":     status,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    push_insight(insight)

    return {
        "disaster_type": dtype,
        "person_count":  pts,
        "car_count":     cts,
        "fire_count":    fts,
        "status":        status
    }

# ─── PUBLIC ENTRYPOINT ─────────────────────────────────────────────────────────
def predict_media(input_url: str, latitude: float, longitude: float) -> dict:
    """
    Download (if URL), then read first frame of image or video,
    analyze, push to Supabase, and return the counts + classification.
    """
    # if it’s a URL, download to temp
    local = download_file(input_url) if input_url.lower().startswith("http") else input_url

    # choose whether image or video by extension
    ext = os.path.splitext(local)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".webp"):
        frame = cv2.imread(local)
        if frame is None:
            raise RuntimeError(f"Cannot read image {local}")
        return analyze_frame(frame, latitude, longitude)

    # else treat as video: open, read one frame
    cap = cv2.VideoCapture(local)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {local}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to grab first frame from video")
    return analyze_frame(frame, latitude, longitude)
