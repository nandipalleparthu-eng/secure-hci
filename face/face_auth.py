"""
Thread-safe face authentication worker based on face_recognition.

Known faces are loaded from data/known_faces. Each image filename becomes the
display name. Gesture control is enabled only when exactly one detected face
matches a known encoding within the configured tolerance.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import face_recognition
import numpy as np


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FaceAuthState:
    name: str = "No Face"
    authorized: bool = False
    face_count: int = 0
    last_update: float = 0.0


class FaceAuthenticator:
    def __init__(
        self,
        known_faces_dir: str = "data/known_faces",
        recognition_interval: int = 6,
        tolerance: float = 0.50,
        resize_scale: float = 0.25,
        state_timeout: float = 1.5,
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
        self._known_encodings: list[np.ndarray] = []
        self._known_names: list[str] = []

        self._load_known_faces()

    def start(self, frame_queue: queue.Queue[np.ndarray]) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(frame_queue,),
            daemon=True,
            name="FaceAuthThread",
        )
        self._thread.start()
        logger.info("Face authentication worker started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_state(self) -> FaceAuthState:
        with self._lock:
            state = FaceAuthState(
                name=self._state.name,
                authorized=self._state.authorized,
                face_count=self._state.face_count,
                last_update=self._state.last_update,
            )

        if state.last_update and (time.time() - state.last_update) > self.state_timeout:
            return FaceAuthState(name="Stale", authorized=False, face_count=0, last_update=state.last_update)
        return state

    def _load_known_faces(self) -> None:
        if not os.path.isdir(self.known_faces_dir):
            logger.warning(
                "Known face directory '%s' not found. Create it and add labeled images.",
                self.known_faces_dir,
            )
            return

        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for file_name in os.listdir(self.known_faces_dir):
            name, ext = os.path.splitext(file_name)
            if ext.lower() not in supported_exts:
                continue

            file_path = os.path.join(self.known_faces_dir, file_name)
            try:
                image = face_recognition.load_image_file(file_path)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) != 1:
                    logger.warning(
                        "Skipping '%s': expected exactly one face, found %s.",
                        file_name,
                        len(encodings),
                    )
                    continue
                self._known_encodings.append(encodings[0])
                self._known_names.append(name)
                logger.info("Loaded face encoding for '%s'.", name)
            except Exception as exc:
                logger.error("Failed to load '%s': %s", file_name, exc)

        logger.info("Known face database size: %s", len(self._known_names))

    def _run(self, frame_queue: queue.Queue[np.ndarray]) -> None:
        while not self._stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self._frame_count += 1
            if self._frame_count % self.recognition_interval != 0:
                continue

            next_state = self._recognize(frame)
            with self._lock:
                self._state = next_state

    def _recognize(self, frame: np.ndarray) -> FaceAuthState:
        small_frame = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small, model="hog")
        if not locations:
            return FaceAuthState(name="No Face", authorized=False, face_count=0, last_update=time.time())

        encodings = face_recognition.face_encodings(rgb_small, locations)
        face_count = len(encodings)
        if face_count != 1:
            return FaceAuthState(
                name="Multiple Faces" if face_count > 1 else "Unknown",
                authorized=False,
                face_count=face_count,
                last_update=time.time(),
            )

        if not self._known_encodings:
            return FaceAuthState(name="Unknown", authorized=False, face_count=face_count, last_update=time.time())

        encoding = encodings[0]
        distances = face_recognition.face_distance(self._known_encodings, encoding)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])

        if best_distance <= self.tolerance:
            return FaceAuthState(
                name=self._known_names[best_index],
                authorized=True,
                face_count=face_count,
                last_update=time.time(),
            )

        return FaceAuthState(name="Unknown", authorized=False, face_count=face_count, last_update=time.time())
