# src/image_mask_detect.py

import os
import cv2
import numpy as np

from detect_predict_mask import detect_and_predict_mask, LABELS

# Colors for each label
COLORS = {
    "with_mask": (0, 255, 0),
    "without_mask": (0, 0, 255),
    "mask_irregular": (0, 255, 255),
}

# Change this to your test image path
IMAGE_PATH = os.path.join("test_images", "sample.jpg")

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise IOError(f"Failed to read image {IMAGE_PATH}")

    locs, preds = detect_and_predict_mask(image)

    for (box, pred) in zip(locs, preds):
        (x, y, w, h) = box
        class_id = np.argmax(pred)
        label = LABELS[class_id]
        color = COLORS.get(label, (255, 255, 255))

        text = f"{label}: {pred[class_id] * 100:.2f}%"
        cv2.putText(image, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Mask Detection - Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
