import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import sys
import os

# Load CNN model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'cnn3d_hand_pose_model_v2_75.h5')
cnn_model = tf.keras.models.load_model(model_path)
CONFIDENCE_THRESHOLD = 0.9

gestures = ["Start_End", "Maju", "Mundur", "Kanan", "Kiri", "Atas", "Bawah", "Putar_kanan", "Putar_kiri", "Undefined"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

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

    gesture_text = "No Hand Detected"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).reshape(1, 21, 3)
            probabilities = cnn_model.predict(landmarks)[0]
            max_prob = np.max(probabilities)
            predicted_class = np.argmax(probabilities)

            if max_prob >= CONFIDENCE_THRESHOLD:
                gesture_text = gestures[predicted_class]

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