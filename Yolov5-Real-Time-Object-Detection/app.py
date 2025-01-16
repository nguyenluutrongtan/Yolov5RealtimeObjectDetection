import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np

import torch
from flask import Flask, render_template, request, redirect, Response

app = Flask(__name__)

# Load Pre-trained Model
model = torch.hub.load(
    "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
)

# Set Model Settings
model.eval()
model.conf = 0.6  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1) 

from io import BytesIO

def gen():
    video_source = os.getenv("VIDEO_SOURCE", "http://192.168.2.27:4747/video")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở nguồn video {video_source}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Lỗi: Không thể đọc khung hình từ nguồn video.")
            break

        # Mã hóa khung hình thành JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # Xử lý khung hình với YOLOv5
        img = Image.open(io.BytesIO(frame_bytes))
        results = model(img, size=640)
        results.print()

        # Vẽ kết quả nhận diện lên khung hình
        img = np.squeeze(results.render())  # RGB
        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR

        # Mã hóa khung hình đã xử lý thành JPEG
        ret, buffer = cv2.imencode('.jpg', img_BGR)
        if not ret:
            continue
        frame_processed = buffer.tobytes()

        # Gửi khung hình đến trình duyệt
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_processed + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
