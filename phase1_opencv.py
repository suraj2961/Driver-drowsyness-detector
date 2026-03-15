import cv2  # OpenCV library for camera and image processing

# Step 1: Connect to webcam
# VideoCapture(0) means "use the default webcam"
# If you have multiple cameras, try 1 or 2
cap = cv2.VideoCapture(0)

# Step 2: Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Step 3: Read webcam continuously in a loop
while True:

    # cap.read() returns two things:
    # ret  → True if frame was read successfully, False if not
    # frame → the actual image (a NumPy array)
    ret, frame = cap.read()

    # If frame wasn't captured properly, skip this iteration
    if not ret:
        print("Failed to grab frame")
        break

    # ── EXPLORE YOUR FRAME ──────────────────────────────────
    # frame.shape gives (height, width, channels)
    # channels = 3 because every pixel has Blue, Green, Red values
    h, w, c = frame.shape
    print(h, w, c)  

    # ── CONVERT COLOR ───────────────────────────────────────
    # MediaPipe needs RGB, but OpenCV gives BGR by default
    # We'll practice the conversion here
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── DRAW ON FRAME ────────────────────────────────────────

    # Draw a green rectangle in the top-left corner
    # cv2.rectangle(image, top-left point, bottom-right point, color BGR, thickness)
    cv2.rectangle(frame, (20, 20), (250, 80), (0, 0, 255), 2)

    # Write "DROWSY" text inside the rectangle
    # cv2.putText(image, text, position, font, size, color BGR, thickness)
    cv2.putText(frame, "DROWSY", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # Show resolution info at bottom of screen
    info_text = f"Resolution: {w} x {h}"
    cv2.putText(frame, info_text, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # ── SHOW THE FRAME ───────────────────────────────────────
    cv2.imshow("Phase 1 - OpenCV Basics", frame)

    # waitKey(1) waits 1 millisecond between frames
    # 0xFF is a bitmask (ignore this for now, just know it's needed)
    # ord('q') is the ASCII code for 'q' key
    # So: if you press Q → break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 4: Always release resources when done
cap.release()                # free the webcam
cv2.destroyAllWindows()      # close all OpenCV windows