from flask import Flask, request, jsonify
import os, tempfile, requests
from werkzeug.utils import secure_filename
from YOLO_basics import predict_image, predict_video

app = Flask(__name__)

@app.route('/')
def index():
    return "✅ RapidWarn YOLO API"

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    media_url = data.get('url')
    lat       = data.get('latitude')
    lon       = data.get('longitude')

    if not media_url:
        return jsonify({"error": "Missing 'url'"}), 400
    if lat is None or lon is None:
        return jsonify({"error": "Missing 'latitude' or 'longitude'"}), 400

    try:
        resp = requests.get(media_url, stream=True)
        resp.raise_for_status()
        ctype = resp.headers.get('Content-Type', '')
        suffix = ".mp4" if "video" in ctype or ".mp4" in media_url else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in resp.iter_content(8192):
                tmp.write(chunk)
            local_path = tmp.name

        if suffix == ".mp4":
            res = predict_video(
                input_source=local_path,
                latitude=lat, longitude=lon
            )
        else:
            res = predict_image(
                image_path=local_path,
                latitude=lat, longitude=lon
            )

        return jsonify({"result":"processed", "insight":res})

    except requests.HTTPError as he:
        return jsonify({"error": str(he)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_and_analyze():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error":"No file uploaded"}), 400

    lat = request.form.get('latitude', type=float)
    lon = request.form.get('longitude', type=float)

    filename = secure_filename(file.filename)
    tmp_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(tmp_path)

    if filename.lower().endswith((".mp4",)):
        res = predict_video(input_source=tmp_path, latitude=lat, longitude=lon)
    else:
        res = predict_image(image_path=tmp_path, latitude=lat, longitude=lon)

    return jsonify({"result":"processed", "insight":res})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
