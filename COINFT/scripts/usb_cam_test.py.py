import cv2
import time

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened")
    exit()

# 🔧 Reduce camera resolution at source (VERY important)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 30)

# 🔧 Reduce internal buffering (helps latency)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("camera", 800, 450)

print("Press q to quit")

prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame failed")
        break

    # OPTIONAL: remove resize entirely since we set camera to 640x360
    # frame = cv2.resize(frame, (640, 360))

    cv2.imshow("camera", frame)

    # FPS counter
    frame_count += 1
    if frame_count % 30 == 0:
        now = time.time()
        fps = 30 / (now - prev_time)
        print(f"FPS: {fps:.1f}")
        prev_time = now

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
