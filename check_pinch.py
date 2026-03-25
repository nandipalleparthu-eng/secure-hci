import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import sys
sys.path.insert(0, r'c:\Users\urk25cs9024\Downloads\PROJECT')
from gesture.gesture_controller import (
    THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP,
    MIDDLE_MCP, WRIST, _dist
)

VIDEO = r'c:\Users\urk25cs9024\Downloads\test3.mp4'
cap   = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.75, min_tracking_confidence=0.65)

min_pinch = 999
min_frame = 0

for i in range(0, total, 3):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok: continue
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)
    if not res.multi_hand_landmarks: continue
    lm     = res.multi_hand_landmarks[0].landmark
    palm_h = max(abs(lm[WRIST].y - lm[MIDDLE_TIP].y), 1e-3)
    pi = _dist(lm[THUMB_TIP], lm[INDEX_TIP])  / palm_h
    pm = _dist(lm[THUMB_TIP], lm[MIDDLE_TIP]) / palm_h
    pr = _dist(lm[THUMB_TIP], lm[RING_TIP])   / palm_h
    avg = (pi + pm + pr) / 3
    if avg < min_pinch:
        min_pinch = avg
        min_frame = i
        min_vals  = (pi, pm, pr)

cap.release()
hands.close()
print(f"Closest 3-finger pinch found at frame {min_frame}")
print(f"  pinch_i={min_vals[0]:.3f}  pinch_m={min_vals[1]:.3f}  pinch_r={min_vals[2]:.3f}")
print(f"  avg={min_pinch:.3f}  (threshold needed: all < 0.30)")
if min_pinch < 0.30:
    print("  DOUBLE_CLICK would be detected with current threshold!")
else:
    print(f"  Need threshold > {min_pinch:.3f} OR record the gesture more clearly")
    print(f"  Suggested threshold: {min_pinch + 0.05:.2f}")
