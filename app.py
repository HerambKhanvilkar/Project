from flask import Flask, request, jsonify
from YOLO_basics import predict_media
import traceback, os

app = Flask(__name__)

@app.route('/')
def index():
    return "✅ ML API is running"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        url = data.get("url")
        lat = data.get("latitude")
        lon = data.get("longitude")

        if not url or lat is None or lon is None:
            return jsonify({"error": "Missing required fields"}), 400

        result = predict_media(url, lat, lon)
        return jsonify({"result": result})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
