import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to draw hand landmarks
def draw_hand_landmarks(image, hand_landmarks): 
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=hand_landmarks,
        connections=mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
    )

# Open the webcam feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to exit the camera feed.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from the camera.")
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Draw landmarks if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks)

    # Display the frame
    cv2.imshow("MediaPipe Hand Tracking", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
