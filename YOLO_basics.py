import os, tempfile, requests, cv2
from ultralytics import YOLO
from datetime import datetime, timezone

# Minimal global config
MODEL = YOLO("yolov8n.pt")  # small model
PERSON_CLASS = 0
CAR_CLASS    = 2
FIRE_CLASS   = 43

# Supabase
SUPABASE_URL = "https://qnttrmrwrenlsnpwcrkl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFudHRybXJ3cmVubHNucHdjcmtsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzI1NTk4OCwiZXhwIjoyMDY4ODMxOTg4fQ.d20cXxyVbdmgO1F4Dvm4B2UTsJCWD37bReL9C-l1J0k"

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

def insert_insight(insight: dict):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    requests.post(f"{SUPABASE_URL}/rest/v1/insights", headers=headers, json=insight)

def predict_media(url: str, latitude: float, longitude: float):
    video_path = download_file(url)
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"error": "Failed to read frame"}

    # Resize to save RAM
    frame = cv2.resize(frame, (320, 320))

    results = MODEL(frame, conf=0.1, iou=0.4)[0]

    persons = sum(int(cls) == PERSON_CLASS for cls in results.boxes.cls)
    cars = sum(int(cls) == CAR_CLASS for cls in results.boxes.cls)
    fires = sum(int(cls) == FIRE_CLASS for cls in results.boxes.cls)

    dtype = "unknown"
    if persons > 30:
        dtype = "stampede"
    elif fires > 0:
        dtype = "riot"
    elif cars > 0:
        dtype = "accident"

    status = "UNSAFE" if persons > 30 else "SAFE"
    insight = {
        "type": dtype,
        "status": status,
        "latitude": latitude,
        "longitude": longitude,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    insert_insight(insight)
    return {"type": dtype, "persons": persons, "cars": cars, "fires": fires, "status": status}
