"""
Full system test - checks every gesture, face auth, and distance.
"""
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import sys
sys.path.insert(0, r'c:\Users\urk25cs9024\Downloads\PROJECT')

from gesture.gesture_controller import (
    GestureState, _classify,
    INDEX_TIP, INDEX_PIP, INDEX_MCP,
    MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP,
    RING_TIP, RING_PIP, RING_MCP,
    PINKY_TIP, PINKY_PIP, PINKY_MCP,
    THUMB_TIP, THUMB_IP,
    _up, _curled, _thumb_out
)

VIDEO = r'c:\Users\urk25cs9024\Downloads\test3.mp4'

cap   = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_v = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total} frames @ {fps_v:.1f} fps")

hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.75, min_tracking_confidence=0.65
)

# Count gesture occurrences across entire video
gesture_counts = {g: 0 for g in GestureState}
swipe = deque(maxlen=14)
total_frames_with_hand = 0

print("\nScanning entire video for gestures...")
for i in range(0, total, 3):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)
    if res.multi_hand_landmarks:
        total_frames_with_hand += 1
        lm = res.multi_hand_landmarks[0].landmark
        h, w = frame.shape[:2]
        state, _, dist = _classify(lm, swipe, w)
        gesture_counts[state] += 1

cap.release()
hands.close()

print(f"\nFrames with hand detected: {total_frames_with_hand}")
print("\nGesture detection results:")
print("-" * 45)
for g, count in gesture_counts.items():
    pct    = (count / max(total_frames_with_hand, 1)) * 100
    status = "✓ DETECTED" if count > 0 else "✗ NOT FOUND"
    print(f"  {g.name:<20} {count:>5} frames ({pct:4.1f}%)  {status}")

print("\nGestures NOT detected in video:")
missing = [g.name for g, c in gesture_counts.items() if c == 0 and g != GestureState.IDLE]
if missing:
    for m in missing:
        print(f"  ✗ {m}")
else:
    print("  All gestures detected!")

print("\n" + "=" * 45)
print("Face auth test...")
print("=" * 45)
from face.face_auth_lite import FaceAuthenticator, _get_descriptor, _cosine_similarity
auth = FaceAuthenticator(
    known_faces_dir=r'c:\Users\urk25cs9024\Downloads\PROJECT\data\known_faces',
    tolerance=0.82
)
print(f"Known: {auth._known_names}, samples: {[len(s) for s in auth._known_descriptors]}")

cap2  = cv2.VideoCapture(VIDEO)
mesh  = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.4
)
auth_count = 0
face_count  = 0
for i in range(0, min(total, 600), 10):
    cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap2.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    res   = mesh.process(rgb)
    if res.multi_face_landmarks:
        face_count += 1
        desc  = _get_descriptor(res.multi_face_landmarks[0].landmark)
        score = float(np.mean([_cosine_similarity(desc, s) for s in auth._known_descriptors[0]]))
        if score >= 0.82:
            auth_count += 1

cap2.release()
mesh.close()
print(f"Face detected: {face_count} frames")
print(f"Authorized:    {auth_count} frames ({(auth_count/max(face_count,1)*100):.1f}%)")
print("\nDONE")
