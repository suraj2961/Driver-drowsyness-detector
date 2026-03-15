import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── EYE LANDMARK INDICES ─────────────────────────────────────────
#
#        p2(385)  p3(387)
#  p1(362)                p4(263)
#        p5(373)  p6(380)
#
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── MEDIAPIPE SETUP ──────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)
def get_landmark_coords(landmarks, index, frame_w, frame_h):
    """Convert normalized coords to pixel coords"""
    lm = landmarks[index]
    x = int(lm.x * frame_w)
    y = int(lm.y * frame_h)
    return (x, y)


def euclidean_distance(p1, p2):
    """
    Calculate straight-line distance between two points.
    p1, p2 are (x, y) tuples.

    Formula: √((x2-x1)² + (y2-y1)²)
    """
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def calculate_EAR(eye_indices, landmarks, frame_w, frame_h):
    """
    Calculate Eye Aspect Ratio for one eye.

    eye_indices → list of 6 landmark indices [p1,p2,p3,p4,p5,p6]

    EAR formula:
         |p2-p6| + |p3-p5|
        ─────────────────────
             2 × |p1-p4|
    """
    # Get pixel coordinates of all 6 eye points
    p1 = get_landmark_coords(landmarks, eye_indices[0], frame_w, frame_h)
    p2 = get_landmark_coords(landmarks, eye_indices[1], frame_w, frame_h)
    p3 = get_landmark_coords(landmarks, eye_indices[2], frame_w, frame_h)
    p4 = get_landmark_coords(landmarks, eye_indices[3], frame_w, frame_h)
    p5 = get_landmark_coords(landmarks, eye_indices[4], frame_w, frame_h)
    p6 = get_landmark_coords(landmarks, eye_indices[5], frame_w, frame_h)

    # Vertical distances (numerator)
    vertical_1 = euclidean_distance(p2, p6)  # |p2 - p6|
    vertical_2 = euclidean_distance(p3, p5)  # |p3 - p5|

    # Horizontal distance (denominator)
    horizontal = euclidean_distance(p1, p4)  # |p1 - p4|

    # EAR formula
    EAR = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return EAR, [p1, p2, p3, p4, p5, p6]  # return EAR + points for drawing
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:

            # ── CALCULATE EAR FOR BOTH EYES ───────────────────────
            left_EAR,  left_points  = calculate_EAR(LEFT_EYE,  face_landmarks, frame_w, frame_h)
            right_EAR, right_points = calculate_EAR(RIGHT_EYE, face_landmarks, frame_w, frame_h)

            # Average both eyes for final EAR
            # Why average? One eye might blink, averaging is more reliable
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # ── DRAW THE 6 EYE POINTS ────────────────────────────
            for point in left_points + right_points:
                cv2.circle(frame, point, 3, (0, 255, 255), -1)  # yellow dots

            # ── DRAW LINES CONNECTING EYE POINTS ─────────────────
            # This visually shows what EAR is measuring
            # Left eye vertical lines
            cv2.line(frame, left_points[1], left_points[5], (0,255,0), 1)  # p2-p6
            cv2.line(frame, left_points[2], left_points[4], (0,255,0), 1)  # p3-p5
            # Left eye horizontal line
            cv2.line(frame, left_points[0], left_points[3], (0,0,255), 1)  # p1-p4

            # Right eye vertical lines
            cv2.line(frame, right_points[1], right_points[5], (0,255,0), 1)
            cv2.line(frame, right_points[2], right_points[4], (0,255,0), 1)
            # Right eye horizontal line
            cv2.line(frame, right_points[0], right_points[3], (0,0,255), 1)

            # ── DISPLAY EAR VALUES ON SCREEN ─────────────────────
            cv2.putText(frame, f"Left  EAR: {left_EAR:.2f}",  (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Right EAR: {right_EAR:.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Avg   EAR: {avg_EAR:.2f}",   (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ── SIMPLE THRESHOLD TEST ─────────────────────────────
            # This is a PREVIEW of Phase 4 logic
            # We'll build this properly next phase
            if avg_EAR < 0.25:
                cv2.putText(frame, "EYES CLOSED!", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "EYES OPEN", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Phase 3 - EAR Calculation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
