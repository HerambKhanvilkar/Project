from flask import Flask, request, jsonify
import requests, os
from YOLO_basics import predict_media

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ RapidWarn YOLO API is up"

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    url  = data.get('url')
    lat  = data.get('latitude')
    lon  = data.get('longitude')

    # 1) validate
    if not url:
        return jsonify({"error": "Missing 'url'"}), 400
    if lat is None or lon is None:
        return jsonify({"error": "Missing 'latitude' or 'longitude'"}), 400

    # 2) quick HEAD to ensure it’s media
    try:
        h = requests.head(url, timeout=5)
        c = h.headers.get("Content-Type","")
        if h.status_code!=200 or ("text/html" in c):
            return jsonify({"error": "URL not pointing to an image/video"}), 400
    except Exception as e:
        return jsonify({"error": f"Cannot reach URL: {e}"}), 400

    # 3) run your model
    try:
        insight = predict_media(url, lat, lon)
        return jsonify({"result":"processed", "insight":insight})
    except Exception as e:
        app.logger.error(str(e))
        return jsonify({"error":"Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
