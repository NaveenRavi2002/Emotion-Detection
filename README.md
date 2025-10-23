# 😊 Real-Time Emotion Detection Using Streamlit, OpenCV, and CNN

This project is a real-time facial emotion detection web application built using **TensorFlow/Keras**, **OpenCV**, and **Streamlit**. It uses a CNN model trained on facial expression images to predict emotions such as happy, sad, angry, etc., from a live video feed — either from a built-in webcam or a smartphone camera via DroidCam.

---

## 📌 Project Overview

* **Goal**: Detect human emotions from live webcam or mobile camera feed.
* **Interface**: Streamlit-based web UI.
* **Model**: Convolutional Neural Network trained on grayscale facial images (48x48).
* **Face Detection**: OpenCV Haar cascade classifier.
* **Emotion Prediction**: 7-class classification using a pre-trained model.

---

## 🧰 Tools & Libraries Used

| Tool/Library       | Purpose                                     |
| ------------------ | ------------------------------------------- |
| `Streamlit`        | Web app interface                           |
| `OpenCV (cv2)`     | Webcam input, face detection                |
| `NumPy`            | Numerical operations                        |
| `TensorFlow/Keras` | CNN model training and prediction           |
| `Haar Cascade`     | Face detection using OpenCV pre-trained XML |
| `DroidCam`         | Use phone as webcam (USB or WiFi)           |

---

## 🎯 Features

* Real-time webcam or mobile camera support.
* Face detection using Haar cascade.
* Emotion classification into 7 classes.
* Live display of video with overlaid predictions.
* Mobile support using DroidCam via USB (low latency).

---

## 😃 Emotion Classes

```
['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
```

---

## 🧠 Model Architecture

The CNN model used consists of:

* Input: (48, 48, 1) grayscale face image
* Several Conv2D + MaxPooling + Dropout layers (128, 256, 512 filters)
* Flatten layer
* Fully connected Dense layers (512 → 256)
* Output: Dense(7, activation='softmax')

**Loss Function**: `categorical_crossentropy`
**Optimizer**: `adam`
**Evaluation Metric**: `accuracy`

---

## 📊 Model Performance

* Trained for 100 epochs on facial expression dataset
* Final training accuracy \~74%
* Final validation accuracy \~63%
* Consistent learning with low overfitting

---

## 🗃️ Project Structure

```
emotion-detector/
├── emotiondetector.json         # Model architecture (JSON)
├── emotiondetector.h5           # Model weights (HDF5)
├── haarcascade_frontalface_default.xml  # Face detector
├── app.py                       # Streamlit app
├── open webcam using opencv           # OpenCV live demo script
├── README.md                    # Project documentation
```

---

## 🖥️ Code Walkthrough

### 1. Load Model

```python
model_from_json() → loads model structure
model.load_weights() → loads trained weights
```

### 2. Load Face Detector

```python
cv2.CascadeClassifier() → loads Haar XML face detector
```

### 3. Streamlit Web UI

```python
st.title(), st.button(), st.image() → streamlit UI elements
```

### 4. Real-Time Feed Processing

* Reads frames from webcam or IP cam (DroidCam)
* Detects faces using Haar cascade
* Preprocesses image to 48x48 grayscale
* Feeds image to model
* Displays predicted label on screen

### 5. Image Preprocessing

```python
reshape to (1, 48, 48, 1)
normalize by dividing by 255
```

---

## 📱 Using Mobile Camera (DroidCam)

### 🔌 Steps for USB Mode:

1. Install **DroidCam** app on your phone and PC from [dev47apps.com](https://www.dev47apps.com/)
2. Enable **USB Debugging** on phone
3. Connect phone via USB cable
4. Open DroidCam Client → Select USB → Start
5. OpenCV will recognize DroidCam as a virtual webcam (e.g., `cv2.VideoCapture(1)`)


---

## 🚀 Running the App

### A. Run Streamlit Web App:

```bash
streamlit run app.py
```


### B. Install Dependencies:

```bash
pip install streamlit opencv-python tensorflow numpy
```

---

## 🔧 Future Improvements

* Add support for video file uploads
* Switch between front/rear cameras on phone
* Improve FPS and reduce latency further
* Deploy to Streamlit Cloud or HuggingFace Spaces

---

## 👨‍💻 Author

**Naveen Kumar Ravi**
Computer Science Engineering Student | AI & Data Science Enthusiast

---

## ✅ License

This project is open-source and free to use for learning and non-commercial purposes.



