# bismillah

import cv2
import mediapipe as mp
import numpy as np
import os

# Inisialisasi Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Folder Dataset
DATA_PATH = 'CNN_dataset_hand_pose_V4' 

# Daftar Gestur untuk drone
gestures = ["Start_End", "Maju", "Mundur", "Kanan", "Kiri", "Atas", "Bawah", "Putar_kanan", "Putar_kiri"]
num_sequences = 50  
sequence_length = 1  # frame

# Membuat folder untuk menyimpan data
def create_gesture_folders(data_path, gestures, num_sequences):
    for gesture in gestures:
        gesture_path = os.path.join(data_path, gesture)
        if not os.path.exists(gesture_path):
            os.makedirs(gesture_path)
            print(f"Folder '{gesture}' created!")
        
        for sequence_num in range(num_sequences):
            sequence_path = os.path.join(gesture_path, f'sequence_{sequence_num}')
            if not os.path.exists(sequence_path):
                os.makedirs(sequence_path)
                print(f"Folder for sequence {sequence_num} in '{gesture}' created!")

# Menghitung bounding box dari landmark tangan
def calculate_bounding_box(hand_landmarks, image_width, image_height):
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]

    x_min = max(int(min(x_coords) * image_width) - 10, 0)
    x_max = min(int(max(x_coords) * image_width) + 10, image_width)
    y_min = max(int(min(y_coords) * image_height) - 10, 0)
    y_max = min(int(max(y_coords) * image_height) + 10, image_height)

    return x_min, y_min, x_max, y_max

# Menyimpan landmark dan gambar
def save_landmarks_and_images(hand_landmarks, image, gesture, sequence_num, frame_num):
    sequence_path = os.path.join(DATA_PATH, gesture, f'sequence_{sequence_num}')
    
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    npy_file_path = os.path.join(sequence_path, f'frame_{frame_num}.npy')
    np.save(npy_file_path, landmarks)

    # Simpan gambar asli tanpa kerangka
    jpg_file_path = os.path.join(sequence_path, f'frame_{frame_num}.jpg')
    cv2.imwrite(jpg_file_path, image)

    # Crop berdasarkan bounding box tangan
    h, w, _ = image.shape
    x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, w, h)
    cropped_image = image[y_min:y_max, x_min:x_max]

    cropped_file_path = os.path.join(sequence_path, f'frame_{frame_num}-cropped.jpg')
    cv2.imwrite(cropped_file_path, cropped_image)

    print(f"Saved cropped hand image: {cropped_file_path}")

# Fungsi untuk menggambar landmark tangan (visualisasi mesh tangan)
def draw_hand_landmarks(image, hand_landmarks): 
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=hand_landmarks,
        connections=mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
    )

# Fungsi mengumpulkan data
def collect_data_interactive(gesture, num_sequences, sequence_length):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Set lebar jendela
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Set tinggi jendela

    if not cap.isOpened():
        print("Kamera tidak dapat dibuka.")
        return

    for sequence_num in range(num_sequences):
        print(f"Siap untuk mengumpulkan data untuk gesture: {gesture}, sequence: {sequence_num}. Tekan 'f' untuk memulai sequence.")
        
        frame_count = 0
        collecting = False

        while frame_count < sequence_length:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari kamera.")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if collecting:
                        save_landmarks_and_images(hand_landmarks, frame, gesture, sequence_num, frame_count)
                        print(f"Frame {frame_count} captured for sequence {sequence_num} of {gesture}")
                        frame_count += 1

            display_frame = frame.copy()  
            cv2.putText(display_frame, f'{gesture} - Seq {sequence_num} Frame {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if not collecting:
                cv2.putText(display_frame, 'Press "f" to start sequence', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(display_frame, 'Press "ESC" to exit', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Hand Gesture Collection', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('f') and not collecting:
                collecting = True  
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
            
            if collecting and frame_count >= sequence_length:
                break

    cap.release()
    cv2.destroyAllWindows()

# Membuat folder dataset
create_gesture_folders(DATA_PATH, gestures, num_sequences)

# Mengumpulkan data interaktif untuk setiap gesture
for gesture in gestures:
    collect_data_interactive(gesture, num_sequences, sequence_length)

print("Pengumpulan data selesai!")