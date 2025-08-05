import os
import tempfile
import requests
import cv2
from datetime import datetime, timezone
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SUPABASE_URL = "https://qnttrmrwrenlsnpwcrkl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFudHRybXJ3cmVubHNucHdjcmtsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzI1NTk4OCwiZXhwIjoyMDY4ODMxOTg4fQ.d20cXxyVbdmgO1F4Dvm4B2UTsJCWD37bReL9C-l1J0k"

# Load YOLOv8n model
MODEL = YOLO("yolov8n.pt")

# COCO class IDs
PERSON_CLASS = 0
CAR_CLASS    = 2
FIRE_CLASS   = 43  # adjust if using custom dataset

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def download_file(url: str) -> str:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    suffix = os.path.splitext(url)[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return path

def extract_first_frame(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Failed to read first frame")
    return frame

def preprocess_frame(frame, img_size=320):
    resized = cv2.resize(frame, (img_size, img_size))
    blur = cv2.GaussianBlur(resized, (0,0), sigmaX=3, sigmaY=3)
    sharpened = cv2.addWeighted(resized, 1.5, blur, -0.5, 0)
    return sharpened

def _insert_insight(insight: dict):
    url = f"{SUPABASE_URL}/rest/v1/insights"
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation"
    }
    r = requests.post(url, headers=headers, json=insight)
    r.raise_for_status()
    return r.json()

# ─── MAIN LOGIC ─────────────────────────────────────────────────────────────────
def analyze_frame(frame, latitude, longitude):
    inp = preprocess_frame(frame)
    print("🔍 Running model...")
    results = MODEL(inp, conf=0.1, iou=0.45, augment=False)

    persons = sum(int(box.cls)==PERSON_CLASS for res in results for box in res.boxes)
    cars    = sum(int(box.cls)==CAR_CLASS    for res in results for box in res.boxes)
    fires   = sum(int(box.cls)==FIRE_CLASS   for res in results for box in res.boxes)

    if persons > 40:
        dtype = "stampede"
    elif fires > 0:
        dtype = "riot"
    elif cars > 0:
        dtype = "accident"
    else:
        dtype = "unknown"

    status = "SAFE" if persons <= 30 else "UNSAFE"
    insight = {
        "type":       dtype,
        "location":   None,
        "latitude":   latitude,
        "longitude":  longitude,
        "status":     status,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    try:
        _insert_insight(insight)
    except Exception as e:
        print("Supabase insert failed:", e)

    return {
        "disaster_type": dtype,
        "person_count":  persons,
        "car_count":     cars,
        "fire_count":    fires,
        "status":        status
    }

def predict_media(input_source: str, latitude: float, longitude: float):
    print(f"📥 Downloading from {input_source}")
    path = download_file(input_source)
    print(f"✅ File saved to {path}")

    frame = extract_first_frame(path)
    print(f"🖼 Frame size: {frame.shape}")

    return analyze_frame(frame, latitude, longitude)
