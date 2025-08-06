import os, tempfile, requests, cv2
from ultralytics import YOLO
from datetime import datetime, timezone
import psutil

SUPABASE_URL = "https://qnttrmrwrenlsnpwcrkl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFudHRybXJ3cmVubHNucHdjcmtsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzI1NTk4OCwiZXhwIjoyMDY4ODMxOTg4fQ.d20cXxyVbdmgO1F4Dvm4B2UTsJCWD37bReL9C-l1J0k"

MODEL = YOLO("yolov8n.pt")

PERSON_CLASS = 0
CAR_CLASS = 2
FIRE_CLASS = 43  # Adjust if your custom model uses different class id

def download_file(url: str) -> str:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    ext = os.path.splitext(url)[-1]
    suffix = ext if ext else ".mp4"
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
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    r = requests.post(url, headers=headers, json=insight)
    r.raise_for_status()
    return r.json()

def analyze_frame(frame, latitude, longitude, threshold=30):
    # ✅ RAM Usage Before Processing
    process = psutil.Process(os.getpid())
    print("📦 RAM Before Processing (MB):", process.memory_info().rss / 1024 / 1024)

    inp = cv2.resize(frame, (320, 320))
    sharp = sharpen_frame(inp)

    # ✅ RAM Usage Before YOLO Inference
    print("⚙️ RAM Before Inference (MB):", process.memory_info().rss / 1024 / 1024)
    results = MODEL(sharp, conf=0.1, iou=0.45)
    # ✅ RAM Usage After Inference
    print("✅ RAM After Inference (MB):", process.memory_info().rss / 1024 / 1024)

    persons = sum(int(box.cls) == PERSON_CLASS for res in results for box in res.boxes)
    cars = sum(int(box.cls) == CAR_CLASS for res in results for box in res.boxes)
    fires = sum(int(box.cls) == FIRE_CLASS for res in results for box in res.boxes)

    dtype = "stampede" if persons > 40 else "riot" if fires > 0 else "accident" if cars > 0 else "unknown"
    status = "SAFE" if persons <= threshold else "UNSAFE"

    insight = {
        "type": dtype,
        "location": None,
        "latitude": latitude,
        "longitude": longitude,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    try:
        _insert_insight(insight)
    except Exception as e:
        print("Insert failed:", e)

    return {
        "disaster_type": dtype,
        "person_count": persons,
        "car_count": cars,
        "fire_count": fires,
        "status": status
    }

def predict_media(input_source: str, latitude: float, longitude: float):
    path = download_file(input_source)
    if path.lower().endswith((".jpg", ".jpeg", ".png")):
        image = cv2.imread(path)
        return analyze_frame(image, latitude, longitude)
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {"error": f"Can't open video {path}"}

        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {"error": "Can't read frame from video"}

        return analyze_frame(frame, latitude, longitude)
