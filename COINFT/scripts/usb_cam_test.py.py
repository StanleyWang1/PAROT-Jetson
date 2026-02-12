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

    cv2.imshow("Arducam USB", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
