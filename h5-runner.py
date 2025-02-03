import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model_path = 'model\\model_9_lstm_acc100_3.h5'  # Replace with your .h5 model path
model = load_model(model_path)

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define gestures from the dataset
gestures = ["Start_End", "Maju", "Mundur", "Kanan", "Kiri", "Atas", "Bawah", "Putar_kanan", "Putar_kiri"]

# Preprocessing function to extract landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            return landmarks
    return None

# Update the predict_gesture function
sequence = []  # Store a sequence of 25 frames

def predict_gesture(sequence):
    if len(sequence) == 25:  # Ensure the sequence is complete
        input_data = np.expand_dims(sequence, axis=0)  # Reshape to (1, 25, 63)
        prediction = model.predict(input_data, verbose=0)
        gesture_index = np.argmax(prediction)
        confidence = np.max(prediction)
        return gestures[gesture_index], confidence
    return "Collecting...", 0.0

# Update the run_gesture_detection function
def run_gesture_detection():
    global sequence
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Kamera tidak dapat dibuka.")
        return

    frame_count = 0
    start_time = cv2.getTickCount()  # Start time for FPS calculation
    fps = 0.0  # Initialize fps to avoid the UnboundLocalError

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        # Process the frame and extract landmarks
        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            sequence.append(landmarks)  # Append the current frame's landmarks
            if len(sequence) > 25:  # Maintain the sequence length at 25
                sequence.pop(0)

        # Predict gesture
        gesture, confidence = predict_gesture(sequence)

        # FPS calculation
        frame_count += 1
        end_time = cv2.getTickCount()  # End time for FPS calculation
        time_diff = (end_time - start_time) / cv2.getTickFrequency()
        if time_diff >= 1.0:
            fps = frame_count / time_diff
            frame_count = 0
            start_time = end_time

        # Display FPS, predicted gesture, and confidence on the frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Show the frame with landmarks and FPS
        cv2.imshow('Real-time Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the program
run_gesture_detection()
