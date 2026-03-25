"""
Lightweight face authenticator using MediaPipe Face Mesh + OpenCV.
Replaces dlib/face_recognition for cloud deployment.

How it works:
- Extracts 128 facial landmark distances as a face descriptor
- Compares against stored descriptors from known_faces folder
- No heavy dlib models needed — runs on 512MB RAM
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh

# Key landmark indices for a stable face descriptor
DESCRIPTOR_INDICES = [
    1, 4, 5, 6, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54,
    55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93,
    95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 153, 154, 155,
    157, 158, 159, 160, 161, 163, 168, 172, 173, 176, 178, 181, 185, 191,
    234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288,
    291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321,
    323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 380, 381,
    382, 384, 385, 386, 387, 388, 390, 397, 398, 400, 402, 405, 409, 415,
    454, 466, 468, 473,
]


@dataclass(slots=True)
class FaceAuthState:
    name: str = "No Face"
    authorized: bool = False
    face_count: int = 0
    last_update: float = 0.0


def _extract_descriptor(landmarks, w: int, h: int) -> np.ndarray:
    """Extract normalized pairwise distances as face descriptor."""
    pts = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in DESCRIPTOR_INDICES],
        dtype=np.float32,
    )
    # Normalize by face size
    face_size = np.linalg.norm(pts[0] - pts[-1]) + 1e-6
    pts /= face_size
    # Pairwise distances (first 128 pairs)
    desc = []
    for i in range(min(len(pts) - 1, 128)):
        desc.append(float(np.linalg.norm(pts[i] - pts[i + 1])))
    return np.array(desc, dtype=np.float32)


class FaceAuthenticator:
    def __init__(
        self,
        known_faces_dir: str = "data/known_faces",
        recognition_interval: int = 6,
        tolerance: float = 0.18,
        resize_scale: float = 0.5,
        state_timeout: float = 3.0,
    ) -> None:
        self.known_faces_dir = known_faces_dir
        self.recognition_interval = max(1, recognition_interval)
        self.tolerance = tolerance
        self.resize_scale = resize_scale
        self.state_timeout = state_timeout

        self._state = FaceAuthState()
        self._lock = threading.Lock()
        self._frame_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._known_descriptors: list[np.ndarray] = []
        self._known_names: list[str] = []

        self._load_known_faces()

    def start(self, frame_queue: queue.Queue) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(frame_queue,), daemon=True, name="FaceAuthThread"
        )
        self._thread.start()
        logger.info("Face auth worker started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_state(self) -> FaceAuthState:
        with self._lock:
            s = self._state
            state = FaceAuthState(
                name=s.name, authorized=s.authorized,
                face_count=s.face_count, last_update=s.last_update,
            )
        if state.last_update and (time.time() - state.last_update) > self.state_timeout:
            return FaceAuthState(name="Stale", authorized=False, face_count=0, last_update=state.last_update)
        return state

    def _load_known_faces(self) -> None:
        if not os.path.isdir(self.known_faces_dir):
            logger.warning("Known faces dir '%s' not found.", self.known_faces_dir)
            return

        supported = {".jpg", ".jpeg", ".png", ".bmp"}
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
            for fname in os.listdir(self.known_faces_dir):
                name, ext = os.path.splitext(fname)
                if ext.lower() not in supported:
                    continue
                path = os.path.join(self.known_faces_dir, fname)
                img = cv2.imread(path)
                if img is None:
                    logger.warning("Cannot read image: %s", fname)
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = mesh.process(rgb)
                if not result.multi_face_landmarks or len(result.multi_face_landmarks) != 1:
                    logger.warning("Skipping '%s': expected 1 face, found %s.", fname,
                                   len(result.multi_face_landmarks) if result.multi_face_landmarks else 0)
                    continue
                h, w = img.shape[:2]
                desc = _extract_descriptor(result.multi_face_landmarks[0].landmark, w, h)
                self._known_descriptors.append(desc)
                self._known_names.append(name)
                logger.info("Loaded face descriptor for '%s'.", name)

        logger.info("Known faces loaded: %d", len(self._known_names))

    def _run(self, frame_queue: queue.Queue) -> None:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2,
            refine_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as mesh:
            while not self._stop_event.is_set():
                try:
                    frame = frame_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                self._frame_count += 1
                if self._frame_count % self.recognition_interval != 0:
                    continue

                state = self._recognize(frame, mesh)
                with self._lock:
                    self._state = state

    def _recognize(self, frame: np.ndarray, mesh) -> FaceAuthState:
        small = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)

        if not result.multi_face_landmarks:
            return FaceAuthState(name="No Face", authorized=False, face_count=0, last_update=time.time())

        face_count = len(result.multi_face_landmarks)
        if face_count != 1:
            return FaceAuthState(name="Multiple Faces", authorized=False, face_count=face_count, last_update=time.time())

        if not self._known_descriptors:
            return FaceAuthState(name="Unknown", authorized=False, face_count=1, last_update=time.time())

        h, w = small.shape[:2]
        desc = _extract_descriptor(result.multi_face_landmarks[0].landmark, w, h)
        distances = [float(np.linalg.norm(desc - kd)) for kd in self._known_descriptors]
        best_idx = int(np.argmin(distances))
        best_dist = distances[best_idx]

        logger.debug("Best match: %s dist=%.4f threshold=%.4f", self._known_names[best_idx], best_dist, self.tolerance)

        if best_dist <= self.tolerance:
            return FaceAuthState(name=self._known_names[best_idx], authorized=True, face_count=1, last_update=time.time())

        return FaceAuthState(name="Unknown", authorized=False, face_count=1, last_update=time.time())
