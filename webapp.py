"""
Web-based Secure HCI System — Render-ready
- Browser captures webcam via JavaScript
- Sends frames to Flask server via WebSocket
- Server runs MediaPipe face auth + gesture detection
- Browser controls its own mouse via JavaScript
"""

from __future__ import annotations

import base64
import logging
import os
import queue
import socket
import threading
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from PIL import Image

from face.face_auth_lite import FaceAuthenticator, FaceAuthState
from gesture.gesture_controller import GestureController, GestureData, GestureState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "securehci-secret"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=5 * 1024 * 1024,
)

CONFIG = {
    "known_faces_dir": "data/known_faces",
    "face_recognition_interval": 6,
    "face_tolerance": 0.18,
    "face_resize_scale": 0.5,
    "face_state_timeout": 3.0,
    "gesture_detection_confidence": 0.70,
    "gesture_tracking_confidence": 0.60,
}

sessions: dict[str, dict] = {}
sessions_lock = threading.Lock()


def _push(q: queue.Queue, frame: np.ndarray) -> None:
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(frame)
    except queue.Full:
        pass


def get_or_create_session(sid: str) -> dict:
    with sessions_lock:
        if sid not in sessions:
            gesture_result_q: queue.Queue[GestureData] = queue.Queue(maxsize=2)
            gesture_frame_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
            face_frame_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)

            gesture_ctrl = GestureController(
                result_queue=gesture_result_q,
                min_detection_confidence=CONFIG["gesture_detection_confidence"],
                min_tracking_confidence=CONFIG["gesture_tracking_confidence"],
            )
            face_auth = FaceAuthenticator(
                known_faces_dir=CONFIG["known_faces_dir"],
                recognition_interval=CONFIG["face_recognition_interval"],
                tolerance=CONFIG["face_tolerance"],
                resize_scale=CONFIG["face_resize_scale"],
                state_timeout=CONFIG["face_state_timeout"],
            )

            gesture_ctrl.start(gesture_frame_q)
            face_auth.start(face_frame_q)

            sessions[sid] = {
                "gesture_ctrl": gesture_ctrl,
                "face_auth": face_auth,
                "gesture_result_q": gesture_result_q,
                "gesture_frame_q": gesture_frame_q,
                "face_frame_q": face_frame_q,
                "last_gesture": GestureData(),
            }
            logger.info("Session created: %s", sid)
        return sessions[sid]


def destroy_session(sid: str) -> None:
    with sessions_lock:
        if sid in sessions:
            s = sessions.pop(sid)
            s["gesture_ctrl"].stop()
            s["face_auth"].stop()
            logger.info("Session destroyed: %s", sid)


def decode_frame(data_url: str) -> np.ndarray | None:
    try:
        _, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning("Frame decode error: %s", e)
        return None


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def on_connect():
    get_or_create_session(request.sid)
    logger.info("Connected: %s", request.sid)


@socketio.on("disconnect")
def on_disconnect():
    destroy_session(request.sid)
    logger.info("Disconnected: %s", request.sid)


@socketio.on("frame")
def on_frame(data):
    sid = request.sid
    session = get_or_create_session(sid)

    frame = decode_frame(data["image"])
    if frame is None:
        return

    _push(session["gesture_frame_q"], frame.copy())
    _push(session["face_frame_q"], frame.copy())

    try:
        session["last_gesture"] = session["gesture_result_q"].get_nowait()
    except queue.Empty:
        pass

    gesture: GestureData = session["last_gesture"]
    face: FaceAuthState = session["face_auth"].get_state()

    emit("result", {
        "gesture": gesture.state.name,
        "pointer_x": gesture.pointer_x,
        "pointer_y": gesture.pointer_y,
        "scroll_delta": gesture.scroll_delta,
        "hand_present": gesture.hand_present,
        "authorized": face.authorized,
        "user": face.name,
        "face_count": face.face_count,
    })


port = int(os.environ.get("PORT", 5000))

# Required for Railway/Linux headless environment
os.environ.setdefault("DISPLAY", "")
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

if __name__ == "__main__":
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "localhost"
    print(f"\n{'='*52}")
    print(f"  Secure HCI Web App")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}")
    print(f"{'='*52}\n")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
