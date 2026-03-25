import cv2
import numpy as np
import mediapipe as mp
import sys

VIDEO     = r'c:\Users\urk25cs9024\Downloads\test3.mp4'
KNOWN_DIR = r'c:\Users\urk25cs9024\Downloads\PROJECT\data\known_faces'

print("=" * 50)
print("STEP 1: Known face loading")
print("=" * 50)
from face.face_auth_lite import FaceAuthenticator, _get_descriptor, _cosine_similarity
auth = FaceAuthenticator(known_faces_dir=KNOWN_DIR, tolerance=0.82)
print(f"Known faces: {auth._known_names}")
print(f"Samples per person: {[len(s) for s in auth._known_descriptors]}")

print()
print("=" * 50)
print("STEP 2: Face recognition test on video")
print("=" * 50)
cap   = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
mesh  = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.4
)

for i in range(0, min(total, 300), 15):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    res   = mesh.process(rgb)
    if res.multi_face_landmarks:
        desc  = _get_descriptor(res.multi_face_landmarks[0].landmark)
        for i_p, (name, samples) in enumerate(zip(auth._known_names, auth._known_descriptors)):
            score = float(np.mean([_cosine_similarity(desc, s) for s in samples]))
            auth_str = "✓ AUTHORIZED" if score >= 0.82 else "✗ unknown"
            print(f"Frame {i}: '{name}' similarity={score:.4f} {auth_str}")
        break

cap.release()
mesh.close()

print()
print("=" * 50)
print("STEP 3: Hand gesture detection test")
print("=" * 50)
cap   = cv2.VideoCapture(VIDEO)
hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.6, min_tracking_confidence=0.5
)
for i in range(0, min(total, 500), 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)
    if res.multi_hand_landmarks:
        print(f"Frame {i}: Hand detected ✓")
        break
cap.release()
hands.close()
print()
print("DIAGNOSIS COMPLETE")
