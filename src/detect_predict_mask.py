# src/detect_predict_mask.py

import os
import cv2
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = os.path.join("models", "mask_detector_3class.h5")
LB_PATH = os.path.join("models", "label_binarizer.pickle")
FACE_CASCADE_PATH = os.path.join("face_detector", "haarcascade_frontalface_default.xml")

# Load model
print(f"[INFO] loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load label binarizer
print(f"[INFO] loading label binarizer from {LB_PATH} ...")
with open(LB_PATH, "rb") as f:
    lb = pickle.load(f)

LABELS = lb.classes_

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_cascade.empty():
    raise IOError(f"Failed to load Haar Cascade from {FACE_CASCADE_PATH}")

def detect_and_predict_mask(frame):
    """
    Detect faces in a frame and predict mask status.

    Returns:
        locations: list of (x, y, w, h)
        preds: list of prediction arrays
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    faces_list = []
    locations = []

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = preprocess_input(face)

        faces_list.append(face)
        locations.append((x, y, w, h))

    preds = []
    if len(faces_list) > 0:
        faces_array = np.array(faces_list, dtype="float32")
        preds = model.predict(faces_array)

    return locations, preds
