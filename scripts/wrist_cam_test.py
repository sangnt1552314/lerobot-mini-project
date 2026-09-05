import cv2

CAMERA_ID = 0

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_ID}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Press Q to quit")

while True:
    ok, frame = cap.read()

    if not ok:
        print("Failed to read camera")
        break

    cv2.imshow("SO-101 Wrist Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()