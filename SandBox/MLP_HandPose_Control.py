import cv2
import mediapipe as mp
import numpy as np
import joblib
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'mlp_hand_pose_model_V3_75.pkl')
mlp_model = joblib.load(model_path)
CONFIDENCE_THRESHOLD = 0.9

poses = ["Start_End", "Maju", "Mundur", "Kanan", "Kiri", "Atas", "Bawah", "Putar_kanan", "Putar_kiri", "Undefined"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")

prev_time = 0  # Variabel untuk menghitung FPS

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
            # Konversi landmark ke format numpy array (1D flattened)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)

            # Prediksi menggunakan model MLP
            probabilities = mlp_model.predict_proba(landmarks)[0]
            max_prob = np.max(probabilities)
            predicted_class = np.argmax(probabilities)

            if max_prob >= CONFIDENCE_THRESHOLD:
                gesture_text = poses[predicted_class]

            # Gambar landmarks di tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Tampilkan teks prediksi dan FPS di kamera
    cv2.putText(frame, f'Gesture: {gesture_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Hand Gesture Recognition (MLP)", frame)

    # Tekan 'q' untuk keluar dari program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        return None

    return gesture_text
