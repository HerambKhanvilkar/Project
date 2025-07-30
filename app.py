from flask import Flask, request, jsonify
import os, tempfile, requests
from werkzeug.utils import secure_filename
from YOLO_basics import predict_video, predict_image
from datetime import datetime, timezone

app = Flask(__name__)

@app.route('/')
def index():
    return "✅ RapidWarn YOLO API is running."

@app.route('/analyze', methods=['POST'])
def analyze():
    data      = request.get_json(force=True)
    media_url = data.get('url')
    latitude  = data.get('latitude')
    longitude = data.get('longitude')

    if not media_url:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    if latitude is None or longitude is None:
        return jsonify({"error": "Missing 'latitude' or 'longitude'"}), 400

    try:
        # download to temp
        resp = requests.get(media_url, stream=True)
        resp.raise_for_status()
        suffix = os.path.splitext(media_url)[1].lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in resp.iter_content(8192):
                tmp.write(chunk)
            local_path = tmp.name

        # dispatch
        if suffix in (".jpg", ".jpeg", ".png", ".webp"):
            result = predict_image(
                input_path=local_path,
                latitude=latitude,
                longitude=longitude
            )
        else:
            result = predict_video(
                input_source=local_path,
                output_path="output.mp4",
                latitude=latitude,
                longitude=longitude
            )

        return jsonify({
            "result":   "processed",
            "media":    media_url,
            "location": {"latitude": latitude, "longitude": longitude},
            "insight":  result
        })

    except requests.HTTPError as he:
        return jsonify({"error": f"Download failed: {he}"}), 502
    except Exception as e:
        app.logger.error(f"Server error: {e}")
        return jsonify({"error": str(e)}), 500


# ── NEW UPLOAD ENDPOINT ─────────────────────────────────────────────────────────
@app.route('/upload', methods=['POST'])
def upload_and_analyze():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    # save to temp
    filename = secure_filename(file.filename)
    tmp_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(tmp_path)

    # optional GPS
    lat = request.form.get('latitude', type=float)
    lon = request.form.get('longitude', type=float)

    # run prediction
    result = predict_video(
        input_source=tmp_path,
        output_path="output.mp4",
        latitude=lat,
        longitude=lon
    )

    return jsonify({
        "status":  "processed",
        "insight": result
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
