"""Cycle through camera indices and show a live preview of each one, labeled.

Use this to figure out which cv2 index is the wrist cam vs the third-person
cam vs anything else (built-in FaceTime camera, iPhone Continuity Camera, etc).

Run: python camera_id_check.py
Press any key to move to the next index, Q to quit.
"""

import cv2

MAX_INDEX_TO_TRY = 5

for camera_id in range(MAX_INDEX_TO_TRY + 1):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"index {camera_id}: could not open, skipping")
        cap.release()
        continue

    ok, frame = cap.read()
    if not ok:
        print(f"index {camera_id}: opened but failed to read a frame, skipping")
        cap.release()
        continue

    print(f"index {camera_id}: showing preview - press any key for next, Q to quit")
    cv2.putText(frame, f"index {camera_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Camera ID check", frame)
    key = cv2.waitKey(0) & 0xFF
    cap.release()

    if key == ord("q"):
        break

cv2.destroyAllWindows()
