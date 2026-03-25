"""
Test runner for the Secure HCI System using a video file.

Usage:
    python test_video.py --video path/to/your/video.mp4

Controls:
    ESC   - Exit
    F     - Freeze / Unfreeze gesture control
    SPACE - Pause / Resume video playback
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

from face.face_auth_lite import FaceAuthenticator, FaceAuthState
from gesture.gesture_controller import (
    GestureController, GestureData, GestureState,
    INDEX_TIP, INDEX_PIP, INDEX_MCP,
    MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP,
    RING_TIP, RING_PIP, RING_MCP,
    PINKY_TIP, PINKY_PIP, PINKY_MCP,
    THUMB_TIP, THUMB_IP,
    _up, _curled, _thumb_out,
)
from utils.smoothing import CursorSmoother, Debouncer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

CONFIG = {
    "frame_width":  960,
    "frame_height": 540,
    "queue_size":   2,
    "gesture_detection_confidence": 0.75,
    "gesture_tracking_confidence":  0.65,
    "known_faces_dir":        "data/known_faces",
    "face_recognition_interval": 4,
    "face_tolerance":         0.82,
    "face_resize_scale":      0.5,
    "face_state_timeout":     3.0,
    "cursor_alpha":           0.22,
    "click_cooldown":         0.80,
    "scroll_cooldown":        0.10,
    "scroll_strength":        120,
    "window_name":            "Secure HCI System",
}

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()

cam_to_gesture:  queue.Queue[np.ndarray] = queue.Queue(maxsize=CONFIG["queue_size"])
cam_to_face:     queue.Queue[np.ndarray] = queue.Queue(maxsize=CONFIG["queue_size"])
gesture_results: queue.Queue[GestureData] = queue.Queue(maxsize=CONFIG["queue_size"])

display_lock  = threading.Lock()
display_frame: np.ndarray | None = None

landmark_lock  = threading.Lock()
latest_lm      = None          # latest hand landmarks for drawing

stop_event = threading.Event()


@dataclass(slots=True)
class RuntimeState:
    gesture:       GestureData  = field(default_factory=GestureData)
    face:          FaceAuthState = field(default_factory=FaceAuthState)
    manual_freeze: bool          = False
    fps:           float         = 0.0


def push_latest(q: queue.Queue, frame: np.ndarray) -> None:
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(frame)
    except queue.Full:
        pass


# ── Video thread ──────────────────────────────────────────────
def video_thread(video_path: str) -> None:
    global display_frame

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        stop_event.set()
        return

    fps_v       = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps_v
    logger.info("Video: %s  (%.1f fps)", video_path, fps_v)

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CONFIG["frame_width"], CONFIG["frame_height"]))

        with display_lock:
            display_frame = frame.copy()

        push_latest(cam_to_gesture, frame.copy())
        push_latest(cam_to_face,    frame.copy())
        time.sleep(frame_delay)

    cap.release()


# ── Landmark extraction thread ────────────────────────────────
def landmark_thread() -> None:
    """Runs MediaPipe Hands separately just for drawing landmarks."""
    global latest_lm
    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.65,
    ) as hands:
        while not stop_event.is_set():
            with display_lock:
                frame = None if display_frame is None else display_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            with landmark_lock:
                latest_lm = res.multi_hand_landmarks[0] if res.multi_hand_landmarks else None
            time.sleep(0.03)


# ── Overlay drawing ───────────────────────────────────────────
GESTURE_COLORS = {
    GestureState.IDLE:            (180, 180, 180),
    GestureState.MOVE:            (80,  230, 80),
    GestureState.CLICK:           (100, 160, 255),
    GestureState.RIGHT_CLICK:     (100, 100, 255),
    GestureState.DOUBLE_CLICK:    (200, 160, 255),
    GestureState.SCROLL:          (0,   210, 255),
    GestureState.PAUSE:           (0,   80,  220),
    GestureState.DRAG:            (0,   200, 200),
    GestureState.SWITCH_TAB:      (255, 220, 0),
    GestureState.SCREENSHOT:      (255, 255, 255),
    GestureState.ZOOM_IN:         (0,   255, 180),
    GestureState.WORKSPACE_LEFT:  (200, 100, 255),
    GestureState.WORKSPACE_RIGHT: (100, 200, 255),
}


def draw_overlay(frame: np.ndarray, state: RuntimeState) -> np.ndarray:
    h, w = frame.shape[:2]

    # ── Draw hand landmarks ───────────────────────────────────
    with landmark_lock:
        lm_data = latest_lm

    if lm_data is not None:
        mp_draw.draw_landmarks(
            frame, lm_data, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=4),
            mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
        )

        # Finger state badges
        lm = lm_data.landmark
        i_up  = _up(lm[INDEX_TIP],  lm[INDEX_PIP],  lm[INDEX_MCP])
        m_up  = _up(lm[MIDDLE_TIP], lm[MIDDLE_PIP], lm[MIDDLE_MCP])
        r_up  = _up(lm[RING_TIP],   lm[RING_PIP],   lm[RING_MCP])
        p_up  = _up(lm[PINKY_TIP],  lm[PINKY_PIP],  lm[PINKY_MCP])
        th_up = _thumb_out(lm[THUMB_TIP], lm[THUMB_IP], lm[INDEX_MCP])

        fingers = [("👍", th_up), ("☝️", i_up), ("✌️", m_up), ("💍", r_up), ("🤙", p_up)]
        labels  = ["T", "I", "M", "R", "P"]
        up_colors   = [(0,220,80), (80,200,255), (255,180,0), (200,80,255), (255,80,180)]
        down_color  = (30, 30, 80)
        bx = 14
        for idx, (emoji, up) in enumerate(fingers):
            color = up_colors[idx] if up else down_color
            cv2.rectangle(frame, (bx, h - 52), (bx + 34, h - 20), color, -1)
            cv2.rectangle(frame, (bx, h - 52), (bx + 34, h - 20), (255,255,255), 1)
            cv2.putText(frame, labels[idx], (bx + 9, h - 27),
                        cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            bx += 40

    # ── Top status bar ────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 115), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    auth_color   = (60, 210, 60)  if state.face.authorized else (30, 80, 220)
    ctrl_enabled = state.face.authorized and not state.manual_freeze
    ctrl_color   = (40, 220, 80)  if ctrl_enabled else (0, 0, 220)
    g_color      = GESTURE_COLORS.get(state.gesture.state, (180, 180, 180))

    # Left column
    cv2.putText(frame, f"FPS: {state.fps:4.1f}",
                (14, 28),  cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 255, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"User: {state.face.name}",
                (14, 58),  cv2.FONT_HERSHEY_DUPLEX, 0.65, auth_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Auth: {'YES' if state.face.authorized else 'NO'}",
                (14, 88),  cv2.FONT_HERSHEY_DUPLEX, 0.65, auth_color, 1, cv2.LINE_AA)

    # Right column
    cv2.putText(frame, f"Gesture: {state.gesture.state.name}",
                (320, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, g_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {state.face.face_count}",
                (320, 58), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 220, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Control: {'ON' if ctrl_enabled else 'OFF'}",
                (320, 88), cv2.FONT_HERSHEY_DUPLEX, 0.65, ctrl_color, 1, cv2.LINE_AA)

    # Distance — bottom right of status bar
    dist = state.gesture.distance_cm
    dist_m = dist / 100.0
    if dist <= 0:
        d_color = (120, 120, 120)
        d_text  = "Dist: --"
    elif dist_m < 0.25:
        d_color = (0, 0, 255)      # Red — too close
        d_text  = f"Dist: {dist_m:.2f} m (Too Close!)"
    elif dist_m < 0.50:
        d_color = (0, 140, 255)    # Orange
        d_text  = f"Dist: {dist_m:.2f} m (Close)"
    elif dist_m < 0.80:
        d_color = (0, 255, 180)    # Cyan-green — ideal
        d_text  = f"Dist: {dist_m:.2f} m (Ideal)"
    elif dist_m < 1.20:
        d_color = (255, 220, 0)    # Yellow
        d_text  = f"Dist: {dist_m:.2f} m (Far)"
    else:
        d_color = (180, 0, 255)    # Purple — too far
        d_text  = f"Dist: {dist_m:.2f} m (Too Far!)"
    cv2.putText(frame, d_text,
                (700, 58), cv2.FONT_HERSHEY_DUPLEX, 0.65, d_color, 1, cv2.LINE_AA)

    # Hint bar
    cv2.putText(frame, "ESC Exit | F Freeze | SPACE Pause",
                (w - 310, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.48, (120, 120, 120), 1, cv2.LINE_AA)

    return frame


# ── Controls ──────────────────────────────────────────────────
def apply_controls(runtime: RuntimeState, smoother: CursorSmoother,
                   click_db: Debouncer, scroll_db: Debouncer) -> None:
    g = runtime.gesture
    if not runtime.face.authorized or runtime.manual_freeze or not g.hand_present:
        smoother.reset()
        return

    # PAUSE only freezes movement, not all gestures
    if g.state == GestureState.PAUSE:
        smoother.reset()
        return

    if g.state == GestureState.MOVE:
        rx, ry = int(np.clip(g.pointer_x, 0, 1) * SCREEN_W), \
                 int(np.clip(g.pointer_y, 0, 1) * SCREEN_H)
        sx, sy = smoother.smooth(rx, ry)
        pyautogui.moveTo(int(sx), int(sy))

    elif g.state == GestureState.CLICK and click_db.is_ready():
        pyautogui.click(); click_db.trigger()

    elif g.state == GestureState.RIGHT_CLICK and click_db.is_ready():
        pyautogui.rightClick(); click_db.trigger()

    elif g.state == GestureState.DOUBLE_CLICK and click_db.is_ready():
        pyautogui.doubleClick(); click_db.trigger()

    elif g.state == GestureState.DRAG:
        rx, ry = int(np.clip(g.pointer_x, 0, 1) * SCREEN_W), \
                 int(np.clip(g.pointer_y, 0, 1) * SCREEN_H)
        sx, sy = smoother.smooth(rx, ry)
        pyautogui.dragTo(int(sx), int(sy), button='left', duration=0.05)

    elif g.state == GestureState.SCROLL and scroll_db.is_ready():
        amt = int(np.clip(g.scroll_delta * CONFIG["scroll_strength"], -500, 500))
        if amt: pyautogui.scroll(amt); scroll_db.trigger()

    elif g.state == GestureState.SWITCH_TAB and click_db.is_ready():
        pyautogui.hotkey('ctrl', 'tab'); click_db.trigger()

    elif g.state == GestureState.SCREENSHOT and click_db.is_ready():
        pyautogui.hotkey('win', 'shift', 's'); click_db.trigger()

    elif g.state == GestureState.ZOOM_IN and scroll_db.is_ready():
        pyautogui.hotkey('ctrl', '+'); scroll_db.trigger()

    elif g.state == GestureState.WORKSPACE_LEFT and click_db.is_ready():
        pyautogui.hotkey('ctrl', 'win', 'left'); click_db.trigger()

    elif g.state == GestureState.WORKSPACE_RIGHT and click_db.is_ready():
        pyautogui.hotkey('ctrl', 'win', 'right'); click_db.trigger()


# ── Main ──────────────────────────────────────────────────────
def main(video_path: str) -> None:
    runtime = RuntimeState()

    threads = [
        threading.Thread(target=video_thread,    args=(video_path,), daemon=True, name="VideoThread"),
        threading.Thread(target=landmark_thread, daemon=True, name="LandmarkThread"),
    ]

    gesture_ctrl = GestureController(
        result_queue=gesture_results,
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

    for t in threads:
        t.start()
    gesture_ctrl.start(cam_to_gesture)
    face_auth.start(cam_to_face)

    smoother  = CursorSmoother(alpha=CONFIG["cursor_alpha"])
    click_db  = Debouncer(CONFIG["click_cooldown"])
    scroll_db = Debouncer(CONFIG["scroll_cooldown"])

    cv2.namedWindow(CONFIG["window_name"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONFIG["window_name"], CONFIG["frame_width"], CONFIG["frame_height"])

    fps_frames  = 0
    fps_started = time.time()

    try:
        while not stop_event.is_set():
            try:
                runtime.gesture = gesture_results.get_nowait()
            except queue.Empty:
                pass

            runtime.face = face_auth.get_state()
            apply_controls(runtime, smoother, click_db, scroll_db)

            with display_lock:
                frame = None if display_frame is None else display_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            fps_frames += 1
            elapsed = time.time() - fps_started
            if elapsed >= 1.0:
                runtime.fps = fps_frames / elapsed
                fps_frames  = 0
                fps_started = time.time()

            frame = draw_overlay(frame, runtime)
            cv2.imshow(CONFIG["window_name"], frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord("f"), ord("F")):
                runtime.manual_freeze = not runtime.manual_freeze
                logger.info("Freeze: %s", runtime.manual_freeze)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        gesture_ctrl.stop()
        face_auth.stop()
        cv2.destroyAllWindows()
        logger.info("Shut down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    main(parser.parse_args().video)
