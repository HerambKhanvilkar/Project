# app.py
import traceback
from flask import Flask, request, jsonify
import requests, tempfile, os
from YOLO_basics import predict_video

app = Flask(__name__)

@app.route('/')
def index():
    return "✅ RapidWarn YOLO API is running."

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    url  = data.get('url')
    lat  = data.get('latitude')
    lon  = data.get('longitude')

    if not url:
        return jsonify({"error": "Missing 'url'"}), 400
    if lat is None or lon is None:
        return jsonify({"error": "Missing 'latitude' or 'longitude'"}), 400

    try:
        # simply call our helper
        result = predict_video(input_source=url, latitude=lat, longitude=lon)
        return jsonify({"result": "processed", "insight": result})

    except requests.HTTPError as he:
        return jsonify({"error": f"Download failed: {he}"}), 502

    except Exception:
        tb = traceback.format_exc()
        app.logger.error(f"Internal error:\n{tb}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
