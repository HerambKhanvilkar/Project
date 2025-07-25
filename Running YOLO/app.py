from flask import Flask, request, jsonify
from YOLO_basics import main  # Import your existing logic
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "YOLOv8 Model API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    video_path = data.get('path')  # expects a local path or you can extend for URL

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Invalid path"}), 400

    # Output will be saved to this file
    output_path = "output.mp4"

    # Call your existing main() function
    main(video_path, output_path)

    return jsonify({"result": "Video processed", "output_file": output_path})

if __name__ == "__main__":
    app.run(debug=True)
