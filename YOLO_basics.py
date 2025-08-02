# YOLO_basics.py
import os, tempfile, requests, cv2
from ultralytics import YOLO
from datetime import datetime, timezone

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SUPABASE_URL = "https://qnttrmrwrenlsnpwcrkl.supabase.co"
SUPABASE_KEY = "YOUR_SERVICE_ROLE_KEY"   # ← replace with the service role key from API Keys

# load once
MODEL = YOLO("yolov8n.pt")

# COCO IDs
PERSON_CLASS = 0
CAR_CLASS    = 2
FIRE_CLASS   = 43   # adjust if you have a custom fire class

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

def sharpen_frame(frame):
    blur = cv2.GaussianBlur(frame, (0,0), sigmaX=3, sigmaY=3)
    return cv2.addWeighted(frame, 1.5, blur, -0.5, 0)

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

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def predict_video(
    input_source: str,
    latitude: float = None,
    longitude: float = None,
    threshold: int = 30,
    conf_thresh: float = 0.1,
    img_size: int = 320
) -> dict:
    # get local file
    path = download_file(input_source) if input_source.lower().startswith("http") else input_source

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": f"Cannot open video {path}"}

    # read only the first frame
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"error": "Failed to read first frame"}

    # preprocess
    small = cv2.resize(frame, (img_size, img_size))
    sharp = sharpen_frame(small)

    # inference
    results = MODEL(sharp, conf=conf_thresh, iou=0.45, augment=False)

    # count detections
    persons = sum(int(box.cls)==PERSON_CLASS for res in results for box in res.boxes)
    cars    = sum(int(box.cls)==CAR_CLASS    for res in results for box in res.boxes)
    fires   = sum(int(box.cls)==FIRE_CLASS   for res in results for box in res.boxes)

    # decide disaster
    if persons > 40:
        dtype = "stampede"
    elif fires > 0:
        dtype = "riot"
    elif cars > 0:
        dtype = "accident"
    else:
        dtype = "unknown"

    status = "SAFE" if persons <= threshold else "UNSAFE"
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
