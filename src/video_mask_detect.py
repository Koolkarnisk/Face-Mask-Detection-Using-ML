# src/video_mask_detect.py

import cv2
import numpy as np
from detect_predict_mask import detect_and_predict_mask, LABELS

COLORS = {
    "with_mask": (0, 255, 0),
    "without_mask": (0, 0, 255),
    "mask_irregular": (0, 255, 255)
}

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    locs, preds = detect_and_predict_mask(frame)

    for (box, pred) in zip(locs, preds):
        (x, y, w, h) = box
        class_id = np.argmax(pred)
        label = LABELS[class_id]
        color = COLORS[label]

        text = f"{label}: {pred[class_id]*100:.2f}%"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Mask Detection (Webcam)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
