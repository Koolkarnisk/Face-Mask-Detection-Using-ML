# Face Mask Detection Using Machine Learning

This project implements a **Face Mask Detection System** based on the research paper:

> **"Face Mask Detection Using Machine Learning"**  
> Author: *Sairaj Kulkarni*  
> Published in **IRJMETS** (International Research Journal of Modernization in Engineering, Technology and Science)

The system detects whether a person is:
- ✅ Wearing a mask correctly  
- ❌ Not wearing a mask  
- ⚠ Wearing a mask incorrectly (irregular mask)

---

## 🎯 Classes

The model is trained to classify faces into three classes:

1. `with_mask`
2. `without_mask`
3. `mask_irregular` (nose exposed, chin mask, scarf, hand over face, etc.)

---

## 🧠 Tech Stack

- Python 3
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- OpenCV
- NumPy
- scikit-learn
- imutils

---

## 📂 Project Structure

```text
face-mask-detector-ml/
│
├── dataset/
│   ├── with_mask/
│   ├── without_mask/
│   └── mask_irregular/
│
├── face_detector/
│   └── haarcascade_frontalface_default.xml
│
├── models/
│   └── mask_detector_3class.h5  (generated after training)
│
├── src/
│   ├── mask_detector_trainer.py
│   ├── detect_predict_mask.py
│   ├── image_mask_detect.py
│   └── video_mask_detect.py
│
├── test_images/
│   └── sample.jpg
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
"# Face-Mask-Detection-Using-ML" 
