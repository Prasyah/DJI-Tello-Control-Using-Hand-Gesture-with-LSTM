import importlib
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from djitellopy import Tello
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")


# Daftar model yang tersedia
models = {
    "CNN HandPose Control": "CNN_HandPose_Control",
    "CNN3D HandPose Control": "CNN3D_HandPose_Control",
    "MLP HandPose Control": "MLP_HandPose_Control"
}

tello = Tello()
tello.connect()

# Queue untuk komunikasi antar-thread
gesture_queue = queue.Queue()

# Status drone (untuk mencegah perintah takeoff/landing berulang)
airborne = False

def get_gesture(testrun_module):
    """Fungsi untuk membaca gesture dari kamera dan memasukkannya ke dalam queue."""
    while True:
        gesture = testrun_module.get_gesture()
        if gesture:
            gesture_queue.put(gesture)

def control_tello(testrun_module):
    """Fungsi untuk mengontrol drone secara langsung dari hasil get_gesture(), tanpa menggunakan queue."""
    global airborne

    last_executed_gesture = None  # Menyimpan gesture terakhir yang dieksekusi
    debounce_time = 0.5  # Waktu minimal antara dua eksekusi gesture yang sama (dalam detik)
    last_execution_time = time.time()

    while True:
        gesture = testrun_module.get_gesture()  # Langsung membaca gesture dari kamera

        if gesture:
            # Pastikan gesture tidak dieksekusi berulang kali dalam waktu singkat
            current_time = time.time()
            if gesture == last_executed_gesture and (current_time - last_execution_time < debounce_time):
                continue  # Skip gesture yang sama jika belum melewati debounce_time

            last_executed_gesture = gesture  # Simpan gesture terakhir
            last_execution_time = current_time  # Update waktu terakhir eksekusi gesture

            if gesture == "Start_End":
                if not airborne:
                    tello.takeoff()
                    airborne = True
                else:
                    tello.land()
                    airborne = False
            elif gesture == "Maju":
                tello.send_rc_control(0, 1, 0, 0)
            elif gesture == "Mundur":
                tello.send_rc_control(0, -1, 0, 0)
            elif gesture == "Kanan":
                tello.send_rc_control(1, 0, 0, 0)
            elif gesture == "Kiri":
                tello.send_rc_control(-1, 0, 0, 0)
            elif gesture == "Atas":
                tello.send_rc_control(0, 0, 1, 0)
            elif gesture == "Bawah":
                tello.send_rc_control(0, 0, -1, 0)
            elif gesture == "Putar_kanan":
                tello.send_rc_control(0, 0, 0, 1)
            elif gesture == "Putar_kiri":
                tello.send_rc_control(0, 0, 0, -1)
            else:
                tello.send_rc_control(0, 0, 0, 0)

        time.sleep(0.05)

def start_control():
    selected_model = model_var.get()
    if selected_model not in models:
        messagebox.showerror("Error", "Pilih model yang valid!")
        return
    
    # Import modul yang dipilih
    testrun_file = models[selected_model]
    testrun_module = importlib.import_module(testrun_file)
    messagebox.showinfo("Info", f"Menggunakan model: {selected_model}")
    
    # Menjalankan kontrol drone dalam thread terpisah
    control_thread = threading.Thread(target=control_tello, args=(testrun_module,), daemon=True)
    control_thread.start()

# UI menggunakan Tkinter
root = tk.Tk()
root.title("Drone Control UI")
root.geometry("400x200")

# Label
ttk.Label(root, text="Pilih Model HandPose Control:").pack(pady=10)

# Dropdown menu
model_var = tk.StringVar()
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=list(models.keys()), state="readonly")
model_dropdown.pack(pady=5)
model_dropdown.set("Pilih Model")

# Tombol untuk mulai
start_button = ttk.Button(root, text="Mulai", command=start_control)
start_button.pack(pady=20)

# Menjalankan UI
root.mainloop()
