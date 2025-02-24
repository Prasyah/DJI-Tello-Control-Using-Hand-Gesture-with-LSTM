import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Load CNN model
model_path = "model/cnn3d_hand_pose_model_v2_75.h5"
cnn_model = tf.keras.models.load_model(model_path)

CONFIDENCE_THRESHOLD = 0.9

# Gesture labels
gestures = ["Start_End", "Maju", "Mundur", "Kanan", "Kiri", "Atas", "Bawah", "Putar_kanan", "Putar_kiri", "Undefined"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def process_hand_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).reshape(1, 21, 3)

def predict_pose(landmarks):
    print("Expected model input shape:", cnn_model.input_shape)
    print("Actual input shape:", landmarks.shape)

    # Pastikan input model dalam format float32
    landmarks = landmarks.astype(np.float32)
    
    probabilities = cnn_model.predict(landmarks)[0]
    max_prob = np.max(probabilities)
    predicted_class = np.argmax(probabilities)
    return (gestures[predicted_class], max_prob) if max_prob >= CONFIDENCE_THRESHOLD else ("Undefined", max_prob)


def run_cnn_model_on_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return
    
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = process_hand_landmarks(hand_landmarks)
                pose, confidence = predict_pose(landmarks)
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f'Gesture: {pose} ({confidence:.2f})', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display FPS
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Live Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run inference using camera
run_cnn_model_on_camera()
