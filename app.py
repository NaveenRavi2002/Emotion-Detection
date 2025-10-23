import streamlit as st
import cv2
import numpy as np
import time
from tensorflow.keras.models import model_from_json

# Load the emotion detection model once
@st.cache_resource
def load_model():
    with open("emotiondetector.json", "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights("emotiondetector.h5")
    return model

model = load_model()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Preprocessing for model
def extract_features(image):
    image = np.array(image)
    image = image.reshape(1, 48, 48, 1)
    return image / 255.0

# UI Layout
st.title("😊 Real-Time Emotion Detection")
st.sidebar.header("📷 Controls")

# Initialize camera state
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

# Start/Stop buttons
start_camera = st.sidebar.button("▶️ Start Camera")
stop_camera = st.sidebar.button("⏹️ Stop Camera")

FRAME_WINDOW = st.empty()
status_text = st.sidebar.empty()

# Update state
if start_camera:
    st.session_state.camera_on = True
    status_text.info("📡 Camera started. Press 'Stop Camera' to stop.")

if stop_camera:
    st.session_state.camera_on = False
    status_text.success("✅ Camera stopped.")

# Capture and process video stream
if st.session_state.camera_on:
    cap = cv2.VideoCapture(1)  # Try 0 or 2 if 1 doesn't work

    if not cap.isOpened():
        st.error("❌ Could not open webcam. Check OBS Virtual Camera is running.")
    else:
        prev_time = time.time()

        while st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))

                img = extract_features(roi_gray)
                prediction = model.predict(img, verbose=0)
                label = labels[prediction.argmax()]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # FPS display
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

            # Display in Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(rgb_frame, channels="RGB", use_column_width=True)

        cap.release()
        cv2.destroyAllWindows()
