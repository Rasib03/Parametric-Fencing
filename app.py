import cv2
import numpy as np
import os
import time
import winsound
import threading
import queue
from ultralytics import YOLO
from datetime import datetime
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, send_file
import base64
import json
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import torch

app = Flask(__name__)


# Configuration
class Config:
    MODEL_PATH = "yolo11m-seg.pt"
    ROI_IMAGE_PATH = r"C:\Users\Adic\Desktop\CV New\WIN_20250819_09_09_18_Pro.jpg"
    MASK_PATH = "roi_freehand_mask.npy"
    MIN_CONFIDENCE = 0.5
    ALERT_FREQ = 1500
    ALERT_DURATION = 1000
    INTRUSION_DELAY = 2
    TARGET_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "truck",
        "bird", "cat", "dog", "horse", "sheep", "cow"
    ]
    CAMERA_INDICES = [0]
    ANOMALY_FOLDER = "anomalies"
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    SHM_NAME = "runway_frame"


config = Config()
if not os.path.exists(config.ANOMALY_FOLDER):
    os.makedirs(config.ANOMALY_FOLDER)


# Global state
class AppState:
    def __init__(self):
        self.detection_active = False
        self.cap = None
        self.roi_ready = os.path.exists(config.MASK_PATH)
        self.last_intrusion_time = 0
        self.intrusion_count = 0
        self.alert_count = 0
        self.screenshots = []
        self.frame_queue = queue.Queue(maxsize=2)
        self.roi_image_base64 = None
        self.model = None
        self.roi_mask = None
        self.lock = threading.Lock()
        self.camera_index = 0
        self.shm = None
        self.camera_proc = None
        self.shm_size = config.FRAME_HEIGHT * config.FRAME_WIDTH * 3
        self.frame_shape = (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3)


state = AppState()


# Helper to unlink stale shared memory
def _unlink_stale(name):
    try:
        shm = SharedMemory(name=name, create=False)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


# Camera worker process
def camera_worker(shm_name, shape, size, camera_index):
    import cv2
    import numpy as np
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(name=shm_name, create=False, size=size)
    buf = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, shape[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, shape[0])
    if not cap.isOpened():
        print("Camera open failed")
        return

    print("Camera started")
    try:
        while True:
            ok, frame = cap.read()
            if ok and frame.shape == shape:
                np.copyto(buf, frame)
            time.sleep(0.01)  # Small sleep to avoid max CPU in loop
    finally:
        cap.release()
        shm.close()


# Initialize model
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state.model = YOLO(config.MODEL_PATH).to(device)
    print(f"Model loaded on {device}")
except Exception as e:
    print(f"Could not load YOLO model: {e}")

# Initialize ROI image
if os.path.exists(config.ROI_IMAGE_PATH):
    img = cv2.imread(config.ROI_IMAGE_PATH)
    _, buffer = cv2.imencode('.jpg', img)
    state.roi_image_base64 = base64.b64encode(buffer).decode('utf-8')

# Load ROI mask if exists
if state.roi_ready:
    roi_mask = np.load(config.MASK_PATH)
    if roi_mask.shape != (config.FRAME_HEIGHT, config.FRAME_WIDTH):
        roi_mask = cv2.resize(roi_mask, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
    state.roi_mask = roi_mask
    print("ROI mask loaded")


def detect_cameras():
    indices = []
    max_cameras_to_check = 10
    for index in range(max_cameras_to_check):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            indices.append(index)
            cap.release()
    return indices if indices else [0]


# In app.py, modify generate_frames
def generate_frames():
    while True:
        try:
            frame_bytes, fps = state.frame_queue.get(timeout=1)
            yield (f'--frame\r\nContent-Type: image/jpeg\r\nX-FPS: {fps:.1f}\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            pass


def play_alert():
    winsound.Beep(config.ALERT_FREQ, config.ALERT_DURATION)


def detection_loop():
    print("Detection loop started")
    while state.detection_active:
        with state.lock:
            if state.shm is None:
                time.sleep(0.1)
                continue

            start_time = time.time()
            # Get frame from shared memory
            raw_frame = np.ndarray(state.frame_shape, dtype=np.uint8, buffer=state.shm.buf)
            frame = np.ascontiguousarray(raw_frame.copy())

            results = state.model(frame, verbose=False)[0]
            overlay = frame.copy()
            alert_triggered = False

            if state.roi_mask is not None:
                color_mask = np.zeros_like(frame)
                color_mask[state.roi_mask == 255] = (0, 255, 255)
                overlay = cv2.addWeighted(overlay, 1, color_mask, 0.3, 0)

            for result in results:
                if len(result.boxes.conf) == 0:
                    continue
                conf = float(result.boxes.conf[0])
                if conf < config.MIN_CONFIDENCE:
                    continue

                label = state.model.names[int(result.boxes.cls[0])].lower()
                if label not in config.TARGET_CLASSES:
                    continue

                # Process segmentation mask
                mask = result.masks[0].data[0].cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask = (mask > 0.5).astype(np.uint8) * 255

                # Check ROI intrusion
                if state.roi_mask is not None and cv2.countNonZero(cv2.bitwise_and(state.roi_mask, mask)) > 0:
                    alert_triggered = True
                    color = (0, 0, 255)
                    state.intrusion_count += 1
                    if state.intrusion_count % 16 == 0:
                        capture_screenshot(frame, label)
                else:
                    color = (0, 255, 0)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label} {conf:.2f}"
                cv2.putText(overlay, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if alert_triggered:
                threading.Thread(target=play_alert, daemon=True).start()
                state.alert_count += 1

            fps = 1.0 / (time.time() - start_time)

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', overlay)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            try:
                if state.frame_queue.full():
                    state.frame_queue.get_nowait()
                state.frame_queue.put_nowait((frame_bytes, fps))
            except queue.Empty:
                pass


def capture_screenshot(frame, label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(config.ANOMALY_FOLDER, f"screenshot_{timestamp}_{label}.jpg")
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ret:
        with open(filename, 'wb') as f:
            f.write(buffer)
    state.screenshots.append((filename, timestamp, label))
    if len(state.screenshots) > 20:
        state.screenshots.pop(0)


def generate_frames():
    print("Frame generator started")
    while True:
        try:
            frame_bytes, fps = state.frame_queue.get(timeout=1)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            pass


@app.route('/')
def index():
    status = "Detection running" if state.detection_active else "Ready"
    if state.roi_ready:
        status += " | ROI set"
    return render_template('index.html',
                           status=status,
                           roi_image=state.roi_image_base64,
                           intrusion_count=state.intrusion_count,
                           alert_count=state.alert_count,
                           min_confidence=config.MIN_CONFIDENCE,
                           camera_index=state.camera_index,
                           camera_indices=config.CAMERA_INDICES,
                           screenshots=state.screenshots[-5:])


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    if state.detection_active:
        stop_detection()
    else:
        start_detection()
    return redirect(url_for('index'))


@app.route('/start_detection')
def start_detection():
    if not state.roi_ready:
        print("ROI not set")
        return "ROI not set", 400

    if state.model is None:
        print("Model not loaded")
        return "Model not loaded", 400

    with state.lock:
        if not state.detection_active:
            _unlink_stale(config.SHM_NAME)
            state.shm = SharedMemory(name=config.SHM_NAME, create=True, size=state.shm_size)
            state.camera_proc = mp.Process(target=camera_worker,
                                           args=(config.SHM_NAME, state.frame_shape, state.shm_size,
                                                 state.camera_index))
            state.camera_proc.start()

            state.detection_active = True
            state.intrusion_count = 0
            state.alert_count = 0

            print("Starting detection thread...")
            threading.Thread(target=detection_loop, daemon=True).start()

    return redirect(url_for('index'))


@app.route('/stop_detection')
def stop_detection():
    with state.lock:
        if state.detection_active:
            state.detection_active = False
            if state.camera_proc:
                state.camera_proc.terminate()
                state.camera_proc.join()
                state.camera_proc = None
            if state.shm:
                state.shm.close()
                state.shm.unlink()
                state.shm = None
            while not state.frame_queue.empty():
                state.frame_queue.get()
    return redirect(url_for('index'))


@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Capture an image from the camera and set it as the ROI image"""
    try:
        if state.detection_active and state.shm is not None:
            # Use the shared memory frame if detection is active
            with state.lock:
                frame = np.ndarray(state.frame_shape, dtype=np.uint8, buffer=state.shm.buf).copy()
        else:
            # Create a temporary camera capture
            cap = cv2.VideoCapture(state.camera_index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

            if not cap.isOpened():
                return "Failed to open camera", 400

            # Allow camera to warm up
            time.sleep(0.5)

            # Read a few frames to allow the camera to adjust
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    return "Failed to capture image", 400
                time.sleep(0.05)  # Small delay between frames

            # Capture one more frame to use as the ROI image
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return "Failed to capture image", 400

        # Save the captured image as the ROI image
        cv2.imwrite(config.ROI_IMAGE_PATH, frame)

        # Update the base64 encoded image for the UI
        _, buffer = cv2.imencode('.jpg', frame)
        state.roi_image_base64 = base64.b64encode(buffer).decode('utf-8')

        print("Image captured and set as ROI image")
        return redirect(url_for('index'))

    except Exception as e:
        print(f"Error capturing image: {e}")
        return f"Error capturing image: {e}", 500


@app.route('/save_roi', methods=['POST'])
def save_roi():
    data = request.json
    points = data.get('points', [])

    if not points or len(points) < 3:
        return "Not enough points", 400

    img = cv2.imread(config.ROI_IMAGE_PATH)
    if img is None:
        return "ROI image not found", 400

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    scaled_points = np.array([(p['x'], p['y']) for p in points], dtype=np.int32)
    cv2.fillPoly(mask, [scaled_points], 255)

    np.save(config.MASK_PATH, mask)
    state.roi_mask = mask
    state.roi_ready = True
    print("ROI saved")

    return "ROI saved", 200


@app.route('/clear_roi')
def clear_roi():
    if os.path.exists(config.MASK_PATH):
        os.remove(config.MASK_PATH)
    state.roi_ready = False
    state.roi_mask = None
    return redirect(url_for('index'))


@app.route('/update_settings', methods=['POST'])
def update_settings():
    min_confidence = float(request.form.get('min_confidence', config.MIN_CONFIDENCE))
    config.MIN_CONFIDENCE = max(0.1, min(min_confidence, 0.9))

    camera_index = int(request.form.get('camera_index', state.camera_index))
    state.camera_index = camera_index

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        file.save(config.ROI_IMAGE_PATH)
        img = cv2.imread(config.ROI_IMAGE_PATH)
        if img.shape[:2] != (config.FRAME_HEIGHT, config.FRAME_WIDTH):
            img = cv2.resize(img, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            cv2.imwrite(config.ROI_IMAGE_PATH, img)
        _, buffer = cv2.imencode('.jpg', img)
        state.roi_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return redirect(url_for('index'))


@app.route('/anomalies')
def anomalies():
    return render_template('anomalies.html', screenshots=state.screenshots)


@app.route('/download_anomaly/<path:filename>')
def download_anomaly(filename):
    return send_from_directory(config.ANOMALY_FOLDER,
                               os.path.basename(filename),
                               as_attachment=True)


if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    config.CAMERA_INDICES = detect_cameras()
    print("Available cameras:", config.CAMERA_INDICES)
    app.run(debug=True, threaded=True, port=5000)