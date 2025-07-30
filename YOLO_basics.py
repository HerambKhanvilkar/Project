import os
import tempfile
import requests
import cv2
from ultralytics import YOLO
from datetime import datetime, timezone

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SUPABASE_URL = "https://qnttrmrwrenlsnpwcrkl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFudHRybXJ3cmVubHNucHdjcmtsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzI1NTk4OCwiZXhwIjoyMDY4ODMxOTg4fQ.d20cXxyVbdmgO1F4Dvm4B2UTsJCWD37bReL9C-l1J0k"  # your service_role key

# COCO class IDs
PERSON_CLASS = 0
CAR_CLASS    = 2
# If you have a fire/smoke class in your custom model, put its index here:
FIRE_CLASS   = 43  # replace with your custom fire/smoke class ID

# ── HELPERS ────────────────────────────────────────────────────────────────────
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
    blurred = cv2.GaussianBlur(frame, (0,0), sigmaX=3, sigmaY=3)
    return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

def _insert_insight(insight: dict):
    """POST a new row into Supabase 'insights' table via REST."""
    url = f"{SUPABASE_URL}/rest/v1/insights"
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation"
    }
    r = requests.post(url, headers=headers, json=insight)
    print(f"Insert status: {r.status_code}, response: {r.text}")
    r.raise_for_status()
    return r.json()

# ── MAIN PROCESSING ────────────────────────────────────────────────────────────
def predict_video(
    input_source: str,
    threshold:    int    = 30,
    conf_thresh:  float  = 0.1,
    img_size:     int    = 1024,
    up_scale:     float  = 1.2,
    output_path:  str    = "output.mp4",
    latitude:     float  = None,
    longitude:    float  = None
) -> dict:
    # fetch or use local file
    if input_source.lower().startswith("http"):
        input_path = download_file(input_source)
    else:
        input_path = input_source

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video {input_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    writer       = None

    model       = YOLO("yolov8n.pt")
    max_persons = 0
    car_count   = 0
    fire_count  = 0
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # preprocess & inference
        sharp = sharpen_frame(frame)
        if up_scale != 1.0:
            sharp = cv2.resize(sharp, None, fx=up_scale, fy=up_scale, interpolation=cv2.INTER_LINEAR)
        results = model(sharp, conf=conf_thresh, iou=0.45, imgsz=img_size, augment=True)

        # init writer
        if writer is None:
            h, w   = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # count detections
        per_frame_persons = 0
        for res in results:
            for box in res.boxes:
                cls = int(box.cls)
                if cls == PERSON_CLASS:
                    per_frame_persons += 1
                elif cls == CAR_CLASS:
                    car_count += 1
                elif cls == FIRE_CLASS:
                    fire_count += 1

                # draw box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                if up_scale != 1.0:
                    x1, y1, x2, y2 = [c / up_scale for c in (x1, y1, x2, y2)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

        max_persons = max(max_persons, per_frame_persons)

        # overlay
        status = "SAFE" if per_frame_persons <= threshold else "UNSAFE"
        color  = (0,255,0) if status == "SAFE" else (0,0,255)
        cv2.putText(frame, f"Count: {per_frame_persons}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, status, (20,80), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
        writer.write(frame)

    cap.release()
    writer.release()

    # classify disaster type
    if max_persons > 40:
        disaster_type = "stampede"
    elif fire_count > 0:
        disaster_type = "riot"
    elif car_count > 0:
        disaster_type = "accident"
    else:
        disaster_type = "unknown"

    insight = {
        "type":       disaster_type,
        "location":   None,
        "latitude":   latitude,
        "longitude":  longitude,
        "status":     "SAFE" if max_persons <= threshold else "UNSAFE",
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    try:
        _insert_insight(insight)
        print("Insight inserted into Supabase.")
    except Exception as e:
        print("Failed to insert insight:", e)

    return {
        "output_path":     output_path,
        "max_count":       max_persons,
        "threshold":       threshold,
        "status":          insight["status"],
        "disaster_type":   disaster_type,
        "processed_frames": frame_idx,
        "total_frames":    total_frames
    }

def predict_image(input_path: str, latitude: float = None, longitude: float = None) -> dict:
    """Placeholder for image-based logic—extend as needed."""
    insight = {
        "type":       "image",
        "location":   None,
        "latitude":   latitude,
        "longitude":  longitude,
        "status":     "UNSAFE" if False else "SAFE",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    try:
        _insert_insight(insight)
    except Exception as e:
        print("Failed to insert insight:", e)
    return {"unsafe_objects_detected": False, **insight}

# ── SMOKE TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    res = predict_video(
        input_source=r"C:\Users\heram\PycharmProjects\PythonProject1\Running YOLO\Saved Pictures\video.mp4",
        threshold=30,
        latitude=19.0760,
        longitude=72.8777
    )
    print(res)
