"""
Shows real-time gesture label on video so you can see what is being detected.
Press ESC to exit.
"""
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import sys

sys.path.insert(0, r'c:\Users\urk25cs9024\Downloads\PROJECT')
from gesture.gesture_controller import GestureState, _classify

VIDEO = r'c:\Users\urk25cs9024\Downloads\test3.mp4'

cap   = cv2.VideoCapture(VIDEO)
hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.75, min_tracking_confidence=0.65
)
draw  = mp.solutions.drawing_utils
swipe = deque(maxlen=14)

COLORS = {
    GestureState.IDLE:            (150, 150, 150),
    GestureState.MOVE:            (0,   230, 0),
    GestureState.CLICK:           (255, 160, 100),
    GestureState.RIGHT_CLICK:     (255, 100, 100),
    GestureState.DOUBLE_CLICK:    (255, 200, 0),
    GestureState.SCROLL:          (0,   200, 255),
    GestureState.PAUSE:           (0,   0,   255),
    GestureState.SWITCH_TAB:      (255, 255, 0),
    GestureState.SCREENSHOT:      (255, 255, 255),
    GestureState.ZOOM_IN:         (0,   255, 200),
    GestureState.WORKSPACE_LEFT:  (200, 100, 255),
    GestureState.WORKSPACE_RIGHT: (100, 200, 255),
    GestureState.DRAG:            (255, 150, 0),
}

while True:
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (960, 540))
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    gesture_name = "NO HAND"
    color        = (100, 100, 100)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        draw.draw_landmarks(frame, res.multi_hand_landmarks[0],
                            mp.solutions.hands.HAND_CONNECTIONS)
        h, w = frame.shape[:2]
        state, _, dist = _classify(lm, swipe, w)
        gesture_name = state.name
        color        = COLORS.get(state, (200, 200, 200))

        # Show finger states for debugging
        from gesture.gesture_controller import (
            INDEX_TIP, INDEX_PIP, INDEX_MCP,
            MIDDLE_TIP, MIDDLE_MCP, RING_TIP, RING_MCP,
            PINKY_TIP, PINKY_MCP, THUMB_TIP, THUMB_IP,
            _up, _curled, _thumb_out
        )
        i_up  = _up(lm[INDEX_TIP],  lm[INDEX_PIP],  lm[INDEX_MCP])
        m_up  = _up(lm[MIDDLE_TIP], lm[10],          lm[MIDDLE_MCP])
        r_up  = _up(lm[RING_TIP],   lm[14],          lm[RING_MCP])
        p_up  = _up(lm[PINKY_TIP],  lm[18],          lm[PINKY_MCP])
        th_up = _thumb_out(lm[THUMB_TIP], lm[THUMB_IP], lm[INDEX_MCP])

        finger_str = f"I:{int(i_up)} M:{int(m_up)} R:{int(r_up)} P:{int(p_up)} T:{int(th_up)}"
        cv2.putText(frame, finger_str, (14, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Dist: {dist:.1f}cm", (14, 160),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 100), 1, cv2.LINE_AA)

    # Big gesture label
    cv2.rectangle(frame, (0, 0), (960, 110), (18, 18, 18), -1)
    cv2.putText(frame, f"GESTURE: {gesture_name}", (14, 70),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "ESC to exit", (800, 100),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

    cv2.imshow("Gesture Debug", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
