from fastapi import FastAPI, Request, HTTPException
from YOLO_basics import analyze_media
import uvicorn
import traceback

app = FastAPI(title="YOLO Disaster Detection Webhook")

@app.get("/")
async def root():
    return {"status": "✅ Webhook is running"}

@app.post("/process")
async def process(request: Request):
    try:
        data = await request.json()
        record = data.get("record", {})
        file_name = record.get("name")
        bucket_id = record.get("bucket_id")

        print("📦 Incoming webhook for:", file_name, "in bucket:", bucket_id)

        if not file_name or not bucket_id:
            raise HTTPException(status_code=400, detail="Invalid webhook payload")

        insight = analyze_media(bucket_id, file_name)
        return {
            "status": "processed",
            "file_name": file_name,
            "bucket_id": bucket_id,
            "insight": insight
        }

    except Exception as e:
        print("❌ ERROR in /process:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
