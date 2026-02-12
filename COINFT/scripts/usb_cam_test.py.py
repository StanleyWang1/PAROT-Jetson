import cv2

cap = cv2.VideoCapture(0)  # try 0 first

if not cap.isOpened():
    print("Camera not opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame failed")
        break
    frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("camera", frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
