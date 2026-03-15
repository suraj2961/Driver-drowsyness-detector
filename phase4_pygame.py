import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
import time

# ── PYGAME ALARM SETUP ───────────────────────────────────────────
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm2.wav")

# ── THRESHOLDS ───────────────────────────────────────────────────
EAR_THRESHOLD   = 0.21   
PITCH_THRESHOLD = -15   
FRAME_THRESHOLD = 20     

# ── COUNTERS & FLAGS ─────────────────────────────────────────────
closed_frame_counter = 0
is_alarm_on          = False
total_drowsy_events  = 0

# ── EYE LANDMARK INDICES ─────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


HEAD_POSE_INDICES = [1, 152, 263, 33, 287, 57]


FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0],    # Nose tip
    [0.0,   -63.6, -12.5],    # Chin
    [-43.3,  32.7, -26.0],    # Left eye
    [43.3,   32.7, -26.0],    # Right eye
    [-28.9, -28.9, -24.1],    # Left mouth
    [28.9,  -28.9, -24.1],    # Right mouth
], dtype=np.float64)

# ── MEDIAPIPE SETUP 
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)


# ── HELPER FUNCTIONS

def get_landmark_coords(landmarks, index, frame_w, frame_h):
    """Convert normalized landmark to pixel coordinates"""
    lm = landmarks[index]
    return (int(lm.x * frame_w), int(lm.y * frame_h))


def euclidean_distance(p1, p2):
    """Straight line distance between two (x,y) points"""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def calculate_EAR(eye_indices, landmarks, frame_w, frame_h):
   
    p1 = get_landmark_coords(landmarks, eye_indices[0], frame_w, frame_h)
    p2 = get_landmark_coords(landmarks, eye_indices[1], frame_w, frame_h)
    p3 = get_landmark_coords(landmarks, eye_indices[2], frame_w, frame_h)
    p4 = get_landmark_coords(landmarks, eye_indices[3], frame_w, frame_h)
    p5 = get_landmark_coords(landmarks, eye_indices[4], frame_w, frame_h)
    p6 = get_landmark_coords(landmarks, eye_indices[5], frame_w, frame_h)

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    EAR = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return EAR, [p1, p2, p3, p4, p5, p6]


def calculate_head_pose(face_landmarks, frame_w, frame_h):
   
    face_2d = []
    for idx in HEAD_POSE_INDICES:
        lm = face_landmarks[idx]
        face_2d.append([lm.x * frame_w, lm.y * frame_h])
    face_2d = np.array(face_2d, dtype=np.float64)

  
    focal_length = frame_w
    cam_matrix = np.array([
        [focal_length, 0,            frame_w / 2],
        [0,            focal_length, frame_h / 2],
        [0,            0,            1          ]
    ], dtype=np.float64)

   
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

   
    success, rotation_vec, translation_vec = cv2.solvePnP(
        FACE_3D_MODEL,
        face_2d,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0, 0, 0

   
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = angles[0]   # up/down
    yaw   = angles[1]  # left/right
    roll  = angles[2]  # tilt

    return pitch, yaw, roll


def draw_eye_lines(frame, pts):
 
    cv2.line(frame, pts[1], pts[5], (0, 255, 0), 1)  # vertical
    cv2.line(frame, pts[2], pts[4], (0, 255, 0), 1)  # vertical
    cv2.line(frame, pts[0], pts[3], (0, 0, 255), 1)  # horizontal
    for pt in pts:
        cv2.circle(frame, pt, 3, (0, 255, 255), -1)


def draw_status_panel(frame, avg_EAR, pitch, counter, events, timer_str):
    

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 90), (280, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

   
    cv2.putText(frame, f"EAR    : {avg_EAR:.3f}", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

   
    pitch_color = (0, 0, 255) if pitch < PITCH_THRESHOLD else (200, 200, 200)
    cv2.putText(frame, f"Pitch  : {pitch:.1f}°", (10, 143),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, pitch_color, 2)

   
    counter_color = (0, 165, 255) if counter > 0 else (200, 200, 200)
    cv2.putText(frame, f"Counter: {counter}/{FRAME_THRESHOLD}", (10, 171),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, counter_color, 2)

   
    cv2.putText(frame, f"Events : {events}", (10, 199),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)

    
    cv2.putText(frame, timer_str, (frame.shape[1] - 190, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


# ── MAIN LOOP ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
session_start = time.time()

print("Drowsiness Detector Started — Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = detector.detect(mp_image)

    # Session timer
    elapsed   = int(time.time() - session_start)
    mins, sec = divmod(elapsed, 60)
    timer_str = f"Session: {mins:02d}:{sec:02d}"

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:

            # ── EAR CALCULATION ───────────────────────────────────
            left_EAR,  left_pts  = calculate_EAR(LEFT_EYE,  face_landmarks, frame_w, frame_h)
            right_EAR, right_pts = calculate_EAR(RIGHT_EYE, face_landmarks, frame_w, frame_h)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # ── HEAD POSE CALCULATION ─────────────────────────────
            pitch, yaw, roll = calculate_head_pose(face_landmarks, frame_w, frame_h)
          

            # ── DRAW EYE LANDMARKS ────────────────────────────────
            draw_eye_lines(frame, left_pts)
            draw_eye_lines(frame, right_pts)

            # ── DROWSINESS DETECTION LOGIC ────────────────────────
            eyes_drowsy = avg_EAR < EAR_THRESHOLD
            head_drowsy = pitch < PITCH_THRESHOLD

            if eyes_drowsy or head_drowsy:
                closed_frame_counter += 1

                # Figure out reason
                if eyes_drowsy and head_drowsy:
                    reason = "EYES + HEAD BOWING!"
                elif eyes_drowsy:
                    reason = "EYES CLOSING!"
                else:
                    reason = "HEAD BOWING!"

                if closed_frame_counter >= FRAME_THRESHOLD:
                    # ── ALERT! ────────────────────────────────────
                    if not is_alarm_on:
                        total_drowsy_events += 1
                        is_alarm_on = True
                        print(f"  Drowsy event #{total_drowsy_events} at {mins:02d}:{sec:02d} — {reason}")

                    if not pygame.mixer.get_busy():
                        alarm_sound.play()

                    # Red alert banner
                    cv2.rectangle(frame, (0, 0), (frame_w, 85), (0, 0, 180), -1)
                    cv2.putText(frame, f" DROWSY! {reason}", (15, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                else:
                    # Warning — building up
                    cv2.putText(frame, f"WARNING: {reason}", (15, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            else:
                # ── AWAKE ─────────────────────────────────────────
                closed_frame_counter = 0
                is_alarm_on          = False
                pygame.mixer.stop()

                cv2.putText(frame, "✓ AWAKE", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Draw info panel
            draw_status_panel(frame, avg_EAR, pitch,
                              closed_frame_counter, total_drowsy_events, timer_str)

    else:
        # No face detected
        cv2.rectangle(frame, (0, 0), (frame_w, 85), (40, 40, 40), -1)
        cv2.putText(frame, "NO FACE DETECTED", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        pygame.mixer.stop()
        closed_frame_counter = 0

    cv2.putText(frame, timer_str, (frame_w - 190, frame_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.imshow("Driver Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── CLEANUP ───────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

print(f"\n Session Summary:")
print(f"   Duration     : {mins:02d}:{sec:02d}")
print(f"   Drowsy Events: {total_drowsy_events}")
print(f"   EAR Threshold: {EAR_THRESHOLD}")
print(f"   Pitch Threshold: {PITCH_THRESHOLD}°")