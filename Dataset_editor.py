import os
import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Path dataset
DATA_PATH = 'dataset_hand_pose_V2'

# Fungsi untuk menghitung bounding box dari landmark tangan
def calculate_bounding_box(hand_landmarks, image_width, image_height):
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]

    x_min = max(int(min(x_coords) * image_width) - 10, 0)
    x_max = min(int(max(x_coords) * image_width) + 10, image_width)
    y_min = max(int(min(y_coords) * image_height) - 10, 0)
    y_max = min(int(max(y_coords) * image_height) + 10, image_height)

    return x_min, y_min, x_max, y_max

# Fungsi untuk mengambil ulang data
def recollect_data_for_sequence(gesture, sequence_num):
    sequence_path = os.path.join(DATA_PATH, gesture, f'sequence_{sequence_num}')
    
    if not os.path.exists(sequence_path):
        os.makedirs(sequence_path)
        print(f"Created missing sequence path: {sequence_path}")

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Kamera tidak dapat dibuka.")
        return

    print(f"Recollecting data for gesture '{gesture}', sequence {sequence_num}. Press 'f' to capture frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Tampilkan frame dengan informasi
                display_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    image=display_frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                cv2.putText(display_frame, 'Press "f" to capture frame', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(display_frame, 'Press "ESC" to exit', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Recollecting Data', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('f') and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Simpan landmark
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                npy_file_path = os.path.join(sequence_path, f'frame_0.npy')
                np.save(npy_file_path, landmarks)

                # Simpan gambar asli
                jpg_file_path = os.path.join(sequence_path, f'frame_0.jpg')
                cv2.imwrite(jpg_file_path, frame)

                # Simpan gambar crop hitam
                h, w, _ = frame.shape
                x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, w, h)
                black_image = np.zeros_like(frame)
                mp_drawing.draw_landmarks(
                    image=black_image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                cropped_black_image = black_image[y_min:y_max, x_min:x_max]
                black_file_path = os.path.join(sequence_path, f'frame_0-black.jpg')
                cv2.imwrite(black_file_path, cropped_black_image)

                print(f"Frame recollected for sequence {sequence_num} of gesture '{gesture}'")
                cap.release()
                cv2.destroyAllWindows()
                return

        elif key == 27:  # ESC to exit
            cap.release()
            cv2.destroyAllWindows()
            return

# Recollect specific sequences for gestures
sequences_to_recollect = {
    "Undefined": [47,48,49]
}

for gesture, sequences in sequences_to_recollect.items():
    for sequence_num in sequences:
        recollect_data_for_sequence(gesture, sequence_num)

print("Recollection of data completed.")
