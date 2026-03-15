import cv2

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()   # read one frame
    cv2.imshow("Webcam Test", frame)  # show it

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press Q to quit
        break

cap.release()
cv2.destroyAllWindows()