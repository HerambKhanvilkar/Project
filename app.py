from flask import Flask, request, jsonify
import traceback
from YOLO_basics import predict_media

app = Flask(__name__)

@app.route('/')
def index():
    return "✅ RapidWarn YOLO API is running."

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json(force=True)
        url = data.get('url')
        lat = data.get('latitude')
        lon = data.get('longitude')

        if not url:
            return jsonify({"error": "Missing 'url'"}), 400
        if lat is None or lon is None:
            return jsonify({"error": "Missing 'latitude' or 'longitude'"}), 400

        print(f"▶️ Received: {url}, lat: {lat}, lon: {lon}")
        result = predict_media(url, lat, lon)
        return jsonify({"result": "processed", "insight": result})

    except Exception as e:
        print("❌ ERROR in /analyze:\n", traceback.format_exc())  # Log full error
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
