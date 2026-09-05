"""Minimal two-camera capture helper using OpenCV."""

import cv2


class CameraManager:
    """Opens a wrist camera and a third-person camera and captures single frames."""

    def __init__(self, wrist_id=0, third_id=1, jpeg_quality=80):
        self.jpeg_quality = jpeg_quality

        self.wrist_cap = cv2.VideoCapture(wrist_id)
        if not self.wrist_cap.isOpened():
            raise RuntimeError(f"Could not open wrist camera (id={wrist_id})")

        self.third_cap = cv2.VideoCapture(third_id)
        if not self.third_cap.isOpened():
            raise RuntimeError(f"Could not open third-person camera (id={third_id})")

    def capture(self):
        """Grabs one frame from each camera. Returns (wrist_frame, third_frame)."""
        ok_wrist, wrist_frame = self.wrist_cap.read()
        if not ok_wrist:
            raise RuntimeError("Failed to read frame from wrist camera")

        ok_third, third_frame = self.third_cap.read()
        if not ok_third:
            raise RuntimeError("Failed to read frame from third-person camera")

        return wrist_frame, third_frame

    def encode_jpeg(self, frame):
        """Encodes a single BGR frame to JPEG bytes."""
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("Failed to JPEG-encode frame")
        return buf.tobytes()

    def close(self):
        self.wrist_cap.release()
        self.third_cap.release()
