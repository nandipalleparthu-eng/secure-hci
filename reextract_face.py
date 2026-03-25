import cv2
import numpy as np
import mediapipe as mp
import sys

VIDEO     = r'c:\Users\urk25cs9024\Downloads\test3.mp4'
SAVE_PATH = r'c:\Users\urk25cs9024\Downloads\PROJECT\data\known_faces\parthu.jpg'

print("Extracting best face frame from test3.mp4...")
cap   = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
mesh  = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.4
)

best_frame = None
best_score = 0

for i in range(0, min(total, 600), 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = mesh.process(rgb)
    if res.multi_face_landmarks and len(res.multi_face_landmarks) == 1:
        lm   = res.multi_face_landmarks[0].landmark
        # Score by face size (bigger = closer = better quality)
        h, w = frame.shape[:2]
        xs   = [l.x * w for l in lm]
        ys   = [l.y * h for l in lm]
        size = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if size > best_score:
            best_score = size
            best_frame = frame.copy()
            best_i     = i

cap.release()
mesh.close()

if best_frame is None:
    print("ERROR: No face found in video!")
    sys.exit(1)

cv2.imwrite(SAVE_PATH, best_frame)
print(f"Saved best face frame (frame {best_i}, score={best_score:.0f}) to parthu.jpg")

# Now test the match distance with new photo
print("\nTesting match distance with new photo...")
from face.face_auth_lite import FaceAuthenticator, _extract_descriptor
auth = FaceAuthenticator(
    known_faces_dir=r'c:\Users\urk25cs9024\Downloads\PROJECT\data\known_faces',
    tolerance=0.30
)

cap  = cv2.VideoCapture(VIDEO)
mesh2 = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.4
)

for i in range(0, min(total, 300), 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    res   = mesh2.process(rgb)
    if res.multi_face_landmarks:
        h, w  = small.shape[:2]
        desc  = _extract_descriptor(res.multi_face_landmarks[0].landmark, w, h)
        dists = [float(np.linalg.norm(desc - kd)) for kd in auth._known_descriptors]
        best  = min(dists)
        print(f"Frame {i}: distance = {best:.4f}")
        if i > 50:
            break

cap.release()
mesh2.close()
print("\nDone! Use the distance values above to set the right tolerance.")
