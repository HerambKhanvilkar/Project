from supabase import create_client, Client
from ultralytics import YOLO
import cv2, tempfile, os, requests
from datetime import datetime, timezone
import traceback
import time

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qnttrmrwrenlsnpwcrkl.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFudHRybXJ3cmVubHNucHdjcmtsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzI1NTk4OCwiZXhwIjoyMDY4ODMxOTg4fQ.d20cXxyVbdmgO1F4Dvm4B2UTsJCWD37bReL9C-l1J0k")
TABLE_NAME = "insights"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load models
base_model = YOLO("yolov8n.pt")  # For person, vehicle, and gun detection
fire_smoke_model = YOLO("models/fire_smoke.pt")  # Detects fire and smoke
accident_model = YOLO("models/accident.pt")  # Detects accident-specific signals

def process_frame(frame):
    persons, cars = 0, 0
    fire_count = smoke_count = gun_count = accident_count = 0

    # Base model
    base_results = base_model(frame, conf=0.25, iou=0.5, verbose=False)
    for result in base_results:
        for box in result.boxes:
            label = base_model.names[int(box.cls.item())]
            if label == "person":
                persons += 1
            elif label in {"car", "truck", "bus"}:
                cars += 1
            elif label == "gun":
                gun_count += 1

    # Fire/Smoke model
    fire_smoke_results = fire_smoke_model(frame, conf=0.3, iou=0.5, verbose=False)
    for result in fire_smoke_results:
        for box in result.boxes:
            label = fire_smoke_model.names[int(box.cls.item())]
            if label == "fire":
                fire_count += 1
            elif label == "smoke":
                smoke_count += 1

    # Accident model
    accident_results = accident_model(frame, conf=0.3, iou=0.5, verbose=False)
    for result in accident_results:
        for box in result.boxes:
            label = accident_model.names[int(box.cls.item())]
            if label == "accident":
                accident_count += 1

    return persons, cars, fire_count, smoke_count, gun_count, accident_count

def analyze_media(bucket_id: str, file_name: str):
    try:
        file_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket_id}/{file_name}"
        print("🔍 file_url:", file_url)

        # Get location from DB row
        res = supabase.table(TABLE_NAME).select("location").match({"media_url": file_url}).execute()
        if not res.data:
            print("⚠️ No matching row found in insights table.")
            return {"error": "No matching row found", "processed": False}
        location = res.data[0].get("location", "unknown location")

        # Download file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1])
        r = requests.get(file_url, stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(1024 * 1024):
            tmp.write(chunk)
        tmp.close()

        # Process with YOLO
        cap = cv2.VideoCapture(tmp.name)
        is_video = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 1
        max_persons = max_cars = max_fire = max_smoke = max_gun = max_accident = 0
        frame_count, sample_rate = 0, 5

        if is_video:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % sample_rate:
                    continue
                persons, cars, fire, smoke, gun, accident = process_frame(frame)
                max_persons = max(max_persons, persons)
                max_cars = max(max_cars, cars)
                max_fire = max(max_fire, fire)
                max_smoke = max(max_smoke, smoke)
                max_gun = max(max_gun, gun)
                max_accident = max(max_accident, accident)
            cap.release()
        else:
            frame = cv2.imread(tmp.name)
            if frame is None:
                raise ValueError("Failed to read image file.")
            persons, cars, fire, smoke, gun, accident = process_frame(frame)
            max_persons, max_cars, max_fire, max_smoke, max_gun, max_accident = persons, cars, fire, smoke, gun, accident

        # Release file handle
        try:
            os.remove(tmp.name)
        except PermissionError:
            print("⚠️ Could not delete temp file immediately. Will retry...")
            time.sleep(1)
            try:
                os.remove(tmp.name)
            except Exception as e:
                print("❌ Still couldn't delete temp file:", e)

        # Classification + description + intensity
        disaster_type = "unknown"
        description = None
        intensity = None

        if max_persons >= 10 and max_cars == 0 and max_fire == 0 and max_smoke == 0 and max_gun == 0:
            disaster_type = "stampede"
            description = (
                f"A large crowd of approximately {max_persons} people detected , "
                "with no vehicles or other disaster signals present — possible stampede risk."
            )
        elif max_cars >= 1 and max_accident >= 1:
            disaster_type = "accident"
            description = (
                f"A road accident scenario , involving {max_cars} vehicle(s) "
                f"and {max_accident} accident signal(s) detected."
            )
        elif max_persons >= 5 and (max_fire + max_smoke + max_gun) >= 1:
            disaster_type = "riot"
            description = (
                f"A scene showing signs of civil unrest , "
                f"with {max_persons} people present, {max_fire} fire source(s), "
                f"{max_smoke} smoke plume(s), and {max_gun} possible weapon(s) detected."
            )

        # Intensity only if disaster detected
        if disaster_type != "unknown":
            raw_score = (
                2 * max_fire +
                2 * max_smoke +
                3 * max_gun +
                1 * max_persons +
                2 * max_accident
            )
            if raw_score >= 7:
                intensity = "high"
            elif raw_score >= 4:
                intensity = "moderate"

        # Update row in Supabase
        supabase.table(TABLE_NAME).update({
            "disaster_type": disaster_type,
            "description": description,
            "intensity": intensity,
            "processed": True,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).match({"media_url": file_url}).execute()

        print("✅ Updated row with:", disaster_type, description, intensity)
        return {
            "disaster_type": disaster_type,
            "description": description,
            "intensity": intensity,
            "processed": True
        }

    except Exception as e:
        print("❌ ERROR in analyze_media:", e)
        traceback.print_exc()
        supabase.table(TABLE_NAME).update({
            "processed": False,
            "disaster_type": "error"
        }).match({"media_url": file_url}).execute()
        return {"error": str(e), "processed": False}
