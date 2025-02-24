import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import sys
import os
import time

# Load Keras model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
h5_model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'cnn_hand_pose_model_V1.h5')
model = tf.keras.models.load_model(h5_model_path, compile=False)
IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.6

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

label_map = {
    0: "Start_End",
    1: "Maju",
    2: "Mundur",
    3: "Kanan",
    4: "Kiri",
    5: "Atas",
    6: "Bawah",
    7: "Putar_kanan",
    8: "Putar_kiri"
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")

prev_time = 0

def get_gesture():
    """Menampilkan kamera live dengan pose tangan serta mengembalikan gestur yang terdeteksi."""
    global prev_time

    ret, frame = cap.read()
    if not ret:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture_text = "Undefined"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w) - 10
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w) + 10
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 10
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h) + 10

            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            gray_hand = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            resized_hand = cv2.resize(gray_hand, IMG_SIZE)
            normalized_hand = resized_hand / 255.0
            input_data = normalized_hand.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1).astype(np.float32)

            predictions = model.predict(input_data, verbose=0)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

            if confidence >= CONFIDENCE_THRESHOLD:
                gesture_text = label_map.get(predicted_class, "Unknown")

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Tampilkan teks prediksi dan FPS di kamera
    cv2.putText(frame, f'Gesture: {gesture_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        return None

    return gesture_text