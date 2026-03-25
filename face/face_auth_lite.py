"""
Lightweight face authenticator using MediaPipe Face Mesh.
Uses normalized landmark coordinates as descriptor with cosine similarity.
Enrolls multiple samples per person for robust matching.
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

# 68 stable landmark indices (forehead, eyes, nose, mouth, jaw)
KEY_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109,
    33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
    159, 160, 161, 246, 362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]


@dataclass(slots=True)
class FaceAuthState:
    name: str        = "No Face"
    authorized: bool = False
    face_count: int  = 0
    last_update: float = 0.0


def _get_descriptor(landmarks) -> np.ndarray:
    """
    Extract a pose-normalized face descriptor using key landmarks.
    Normalizes by centering on nose tip and scaling by inter-eye distance.
    """
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in KEY_LANDMARKS], dtype=np.float32)

    # Normalize: center on nose tip (index 0 in KEY_LANDMARKS = landmark 10)
    nose = pts[0].copy()
    pts -= nose

    # Scale by face width (left eye corner to right eye corner)
    left_eye  = np.array([landmarks[33].x,  landmarks[33].y],  dtype=np.float32) - nose
    right_eye = np.array([landmarks[263].x, landmarks[263].y], dtype=np.float32) - nose
    scale = np.linalg.norm(right_eye - left_eye) + 1e-6
    pts /= scale

    desc = pts.flatten()
    # L2 normalize for cosine similarity via dot product
    norm = np.linalg.norm(desc) + 1e-6
    return desc / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class FaceAuthenticator:
    def __init__(
        self,
        known_faces_dir: str   = "data/known_faces",
        recognition_interval: int = 3,
        tolerance: float       = 0.82,   # cosine similarity threshold (0-1, higher = stricter)
        resize_scale: float    = 0.5,
        state_timeout: float   = 3.0,
        enroll_samples: int    = 5,      # number of frames to sample per known image
    ) -> None:
        self.known_faces_dir    = known_faces_dir
        self.recognition_interval = max(1, recognition_interval)
        self.tolerance          = tolerance
        self.resize_scale       = resize_scale
        self.state_timeout      = state_timeout
        self.enroll_samples     = enroll_samples

        self._state   = FaceAuthState()
        self._lock    = threading.Lock()
        self._frame_count = 0
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None

        # Each person has a list of descriptors (multi-sample)
        self._known_descriptors: list[list[np.ndarray]] = []
        self._known_names: list[str] = []

        self._load_known_faces()

    def start(self, frame_queue: queue.Queue) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(frame_queue,),
            daemon=True, name="FaceAuthThread"
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
            return FaceAuthState(name="Stale", authorized=False,
                                 face_count=0, last_update=state.last_update)
        return state

    def _load_known_faces(self) -> None:
        if not os.path.isdir(self.known_faces_dir):
            logger.warning("Known faces dir '%s' not found.", self.known_faces_dir)
            return

        supported = {".jpg", ".jpeg", ".png", ".bmp"}
        with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.3
        ) as mesh:
            for fname in os.listdir(self.known_faces_dir):
                name, ext = os.path.splitext(fname)
                if ext.lower() not in supported:
                    continue
                path = os.path.join(self.known_faces_dir, fname)
                img  = cv2.imread(path)
                if img is None:
                    logger.warning("Cannot read: %s", fname)
                    continue

                descriptors = []
                # Generate multiple samples with slight augmentation
                for s in range(self.enroll_samples):
                    aug = img.copy()
                    # slight brightness variation
                    aug = cv2.convertScaleAbs(aug, alpha=1.0, beta=(s - 2) * 8)
                    rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
                    res = mesh.process(rgb)
                    if res.multi_face_landmarks:
                        desc = _get_descriptor(res.multi_face_landmarks[0].landmark)
                        descriptors.append(desc)

                if not descriptors:
                    logger.warning("No face found in '%s'.", fname)
                    continue

                self._known_descriptors.append(descriptors)
                self._known_names.append(name)
                logger.info("Enrolled '%s' with %d samples.", name, len(descriptors))

        logger.info("Known faces loaded: %d person(s).", len(self._known_names))

    def _run(self, frame_queue: queue.Queue) -> None:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
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
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res   = mesh.process(rgb)

        if not res.multi_face_landmarks:
            return FaceAuthState(name="No Face", authorized=False,
                                 face_count=0, last_update=time.time())

        face_count = len(res.multi_face_landmarks)
        if face_count > 1:
            return FaceAuthState(name="Multiple Faces", authorized=False,
                                 face_count=face_count, last_update=time.time())

        if not self._known_descriptors:
            return FaceAuthState(name="Unknown", authorized=False,
                                 face_count=1, last_update=time.time())

        desc = _get_descriptor(res.multi_face_landmarks[0].landmark)

        best_name  = "Unknown"
        best_score = -1.0

        for i, samples in enumerate(self._known_descriptors):
            # Average similarity across all enrolled samples
            score = float(np.mean([_cosine_similarity(desc, s) for s in samples]))
            logger.debug("'%s' similarity=%.4f threshold=%.4f",
                         self._known_names[i], score, self.tolerance)
            if score > best_score:
                best_score = score
                best_name  = self._known_names[i]

        authorized = best_score >= self.tolerance
        name       = best_name if authorized else "Unknown"
        return FaceAuthState(name=name, authorized=authorized,
                             face_count=1, last_update=time.time())
