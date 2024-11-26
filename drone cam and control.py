import cv2
import threading
import time
from djitellopy import Tello

# Function to display the camera feed in RGB
def stream_video(drone):
    try:
        drone.streamon()
        while True:
            # Get the frame from the Tello drone camera
            frame = drone.get_frame_read().frame
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize the frame (optional)
            rgb_frame = cv2.resize(rgb_frame, (1080, 720))
            
            # Display the RGB frame
            cv2.imshow("Tello Camera (RGB)", rgb_frame)
            
            # Break if 'q' is pressed or stream is stopped
            if cv2.waitKey(1) & 0xFF == ord('q') or not drone.streamon:
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        drone.streamoff()
        cv2.destroyAllWindows()

# Function to control the Tello drone and show battery percentage
def control_tello(drone):
    try:
        print("Press keys to control the drone:")
        print("t: Takeoff")
        print("l: Land")
        print("w: Forward")
        print("s: Backward")
        print("a: Left")
        print("d: Right")
        print("b: Show Battery Percentage")
        print("q: Quit")

        while True:
            key = input("Enter command: ")
            
            if key == 't':
                drone.takeoff()
            elif key == 'l':
                drone.land()
                break
            elif key == 'w':
                drone.move_forward(50)  # Move forward by 50 cm
                time.sleep(2)
            elif key == 's':
                drone.move_back(50)  # Move backward by 50 cm
                time.sleep(2)
            elif key == 'a':
                drone.move_left(50)  # Move left by 50 cm
                time.sleep(2)
            elif key == 'd':
                drone.move_right(50)  # Move right by 50 cm
                time.sleep(2)
            elif key == 'b':
                # Get and display battery percentage
                battery = drone.get_battery()
                print(f"Current battery level: {battery}%")
            elif key == 'q':
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        drone.land()  # Ensure the drone lands if the loop exits
        drone.end()   # End the connection

# Main function with error handling
if __name__ == "__main__":
    tello = Tello()

    try:
        # Connect to the drone
        tello.connect()

        # Start the camera stream in a separate thread
        video_thread = threading.Thread(target=stream_video, args=(tello,))
        video_thread.start()

        # Start controlling the drone
        control_tello(tello)

        # Wait for the video thread to finish
        video_thread.join()

    except Exception as e:
        print(f"Error: {e}")
        tello.land()  # Ensure the drone lands in case of an error
    finally:
        tello.streamoff()
        tello.end()
