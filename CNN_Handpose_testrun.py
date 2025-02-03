import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp

# Load Keras model with compile=False to avoid unnecessary warnings
h5_model_path = 'model/cnn_hand_pose_model_V1.h5'
model = tf.keras.models.load_model(h5_model_path, compile=False)
IMG_SIZE = (64, 64)  # Image size used for model input
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to consider a prediction valid

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Label mapping for gesture classes
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

# Function to run inference on H5 model using camera
def run_h5_model_on_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box coordinates
                h, w, _ = frame.shape
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

                x_min, x_max = int(x_min) - 10, int(x_max) + 10
                y_min, y_max = int(y_min) - 10, int(y_max) + 10

                # Ensure the bounding box is within image dimensions
                x_min, y_min = max(x_min, 0), max(y_min, 0)
                x_max, y_max = min(x_max, w), min(y_max, h)

                # Crop hand region
                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    continue
                
                # Display the cropped hand in a separate window
                cv2.imshow('Cropped Hand', hand_img)
                
                # Preprocess the hand image
                gray_hand = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                resized_hand = cv2.resize(gray_hand, IMG_SIZE)
                normalized_hand = resized_hand / 255.0
                input_data = normalized_hand.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1).astype(np.float32)

                # Run prediction
                predictions = model.predict(input_data, verbose=0)
                predicted_class = np.argmax(predictions)
                confidence = np.max(predictions)

                # Apply confidence threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    gesture_name = label_map.get(predicted_class, "Unknown")
                    gesture_text = f'{gesture_name} ({confidence:.2f})'
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                else:
                    gesture_text = 'Uncertain'

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the frame with prediction
                try:
                    cv2.putText(frame, f'Gesture: {gesture_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except UnicodeEncodeError:
                    print("[Warning] Encoding issue while displaying text.")

        cv2.imshow('Live Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run inference using camera
run_h5_model_on_camera()
