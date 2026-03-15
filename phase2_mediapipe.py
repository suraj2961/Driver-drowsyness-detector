import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── LANDMARK INDICES ─────────────────────────────────────────────
# Same 6 points per eye as before
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── SETUP NEW MEDIAPIPE TASKS API ────────────────────────────────
# BaseOptions → tells mediapipe WHERE the model file is
base_options = python.BaseOptions(
    model_asset_path='face_landmarker.task'  # path to model file you downloaded
)

# FaceLandmarkerOptions → settings for detection
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,                          # detect max 1 face
    min_face_detection_confidence=0.5,    # 50% sure = detect
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create the detector object
detector = vision.FaceLandmarker.create_from_options(options)

# ── HELPER FUNCTION ──────────────────────────────────────────────
def get_landmark_coords(landmarks, index, frame_w, frame_h):
    """
    Converts normalized landmark (0.0 to 1.0)
    to actual pixel coordinates
    """
    lm = landmarks[index]
    x = int(lm.x * frame_w)
    y = int(lm.y * frame_h)
    return (x, y)

# ── MAIN LOOP ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape

    # Convert frame to MediaPipe Image format
    # New API needs mp.Image instead of raw numpy array
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Run detection on this frame
    result = detector.detect(mp_image)

    # result.face_landmarks → list of faces
    # each face → list of 478 landmark points
    if result.face_landmarks:

        for face_landmarks in result.face_landmarks:

            # ── DRAW ALL LANDMARK DOTS ────────────────────────────
            for i in range(len(face_landmarks)):
                coords = get_landmark_coords(face_landmarks, i, frame_w, frame_h)
                cv2.circle(frame, coords, 1, (0, 255, 0), -1)

            # ── HIGHLIGHT EYE POINTS ──────────────────────────────
            for index in LEFT_EYE:
                coords = get_landmark_coords(face_landmarks, index, frame_w, frame_h)
                cv2.circle(frame, coords, 4, (255, 0, 255), -1)  # pink

            for index in RIGHT_EYE:
                coords = get_landmark_coords(face_landmarks, index, frame_w, frame_h)
                cv2.circle(frame, coords, 4, (255, 0, 255), -1)  # pink

        cv2.putText(frame, "FACE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Phase 2 - Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()