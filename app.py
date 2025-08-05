from flask import Flask, request, jsonify
import traceback
from YOLO_basics import predict_media

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ YOLO Disaster Detection API is running."

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    url = data.get("url")
    lat = data.get("latitude")
    lon = data.get("longitude")

    if not url or lat is None or lon is None:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        insight = predict_media(url, lat, lon)
        return jsonify({"status": "processed", "insight": insight})
    except Exception:
        tb = traceback.format_exc()
        app.logger.error(tb)
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
