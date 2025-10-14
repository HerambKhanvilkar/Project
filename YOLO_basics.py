from supabase import create_client, Client
from ultralytics import YOLO
import cv2, tempfile, os, requests
from datetime import datetime, timezone
import traceback
import time

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "insights"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load models
accident_model = YOLO("models/accident.pt")
stampede_model = YOLO("models/stampede.pt")
fire_model     = YOLO("models/fire_smoke.pt")

def get_confidence(model, frame, conf=0.3, iou=0.5):
    """Run YOLO model and return max confidence score."""
    results = model(frame, conf=conf, iou=iou, verbose=False)
    if results and results[0].boxes:
        return float(results[0].boxes.conf.max().cpu().numpy())
    return 0.0

def analyze_media(bucket_id: str, file_name: str, conf_threshold=0.3):
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

        # Read frame (image only for simplicity; extend to video if needed)
        frame = cv2.imread(tmp.name)
        if frame is None:
            raise ValueError("Failed to read image file.")

        # Get confidences
        acc_conf   = get_confidence(accident_model, frame, conf=conf_threshold)
        stamp_conf = get_confidence(stampede_model, frame, conf=conf_threshold)
        fire_conf  = get_confidence(fire_model, frame, conf=conf_threshold)

        # Classification + description
        if acc_conf >= conf_threshold and acc_conf > stamp_conf and acc_conf > fire_conf:
            disaster_type = "accident"
            description = (
                "The scene shows strong signs of a road accident, "
                "with clear indicators of a collision or crash."
            )

        elif stamp_conf >= conf_threshold and stamp_conf > acc_conf and stamp_conf > fire_conf:
            disaster_type = "stampede"
            description = (
                "The scene suggests a potential stampede, "
                "with dense crowd movement dominating the environment."
            )

        elif fire_conf >= conf_threshold and fire_conf > acc_conf and fire_conf > stamp_conf:
            disaster_type = "fire"
            description = (
                "The scene indicates a fire‑related disaster, "
                "with visible flames or smoke being the dominant signals."
            )

        else:
            disaster_type = "normal"
            description = (
                "No strong disaster indicators are present. "
                "The environment appears stable without clear signs of accident, stampede, or fire."
            )
        # Intensity scoring (based on sum of confidences)
        intensity = None
        if disaster_type != "normal":
            raw_score = acc_conf + stamp_conf + fire_conf
            if raw_score >= 1.5:
                intensity = "high"
            elif raw_score >= 0.8:
                intensity = "moderate"
            else:
                intensity = "low"

        # Clean up temp file
        try:
            os.remove(tmp.name)
        except Exception:
            pass

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
        try:
            supabase.table(TABLE_NAME).update({
                "processed": False,
                "disaster_type": "error"
            }).match({"media_url": file_url}).execute()
        except Exception:
            pass
        return {"error": str(e), "processed": False}
