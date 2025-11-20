    # src/mask_detector_trainer.py

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Hyperparameters
INIT_LR = 1e-4
EPOCHS = 25
BS = 32

DATASET_DIR = "dataset"
MODEL_DIR = "models"

CATEGORIES = ["with_mask", "without_mask", "mask_irregular"]

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

print("[INFO] loading images...")
data = []
labels = []

for category in CATEGORIES:
    category_path = os.path.join(DATASET_DIR, category)
    if not os.path.isdir(category_path):
        print(f"[WARN] directory not found: {category_path}")
        continue

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] could not read image: {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

data = np.array(data, dtype="float32")
labels = np.array(labels)

print(f"[INFO] total images loaded: {len(data)}")

# Encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

print("[INFO] classes:", lb.classes_)

# Split data
(trainX, testX, trainY, testY) = train_test_split(
    data, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load base model
print("[INFO] loading MobileNetV2 base...")
baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# Head model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

# Train model
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

# Evaluate
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
trueIdxs = np.argmax(testY, axis=1)

print(classification_report(trueIdxs, predIdxs, target_names=lb.classes_))

# Save model
model_path = os.path.join(MODEL_DIR, "mask_detector_3class.h5")
print(f"[INFO] saving model to {model_path} ...")
model.save(model_path, save_format="h5")

# Save label binarizer
lb_path = os.path.join(MODEL_DIR, "label_binarizer.pickle")
print(f"[INFO] saving label binarizer to {lb_path} ...")
with open(lb_path, "wb") as f:
    pickle.dump(lb, f)

# Plot training
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plot_path = os.path.join(MODEL_DIR, "training_plot.png")
plt.savefig(plot_path)
print(f"[INFO] training plot saved to {plot_path}")
