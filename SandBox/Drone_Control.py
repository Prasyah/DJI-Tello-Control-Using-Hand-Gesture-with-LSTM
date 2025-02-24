import importlib
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from djitellopy import Tello

# Daftar model yang tersedia
models = {
    "CNN HandPose Control": "CNN_HandPose_Control",
    "CNN3D HandPose Control": "CNN3D_HandPose_Control",
    "MLP HandPose Control": "MLP_HandPose_Control"
}

tello = Tello()
tello.connect()

def start_control():
    selected_model = model_var.get()
    if selected_model not in models:
        messagebox.showerror("Error", "Pilih model yang valid!")
        return
    
    # Import modul yang dipilih
    testrun_file = models[selected_model]
    testrun_module = importlib.import_module(testrun_file)
    messagebox.showinfo("Info", f"Menggunakan model: {selected_model}")
    
    # Fungsi untuk mengontrol drone
    def control_tello():
        airborne = False
        while True:
            gesture = testrun_module.get_gesture()
            
            if gesture == "Start_End":
                if not airborne:
                    tello.takeoff()
                    airborne = True
                else:
                    tello.land()
                    break
            elif gesture == "Maju":
                tello.move_forward(50)
            elif gesture == "Mundur":
                tello.move_back(50)
            elif gesture == "Kanan":
                tello.move_right(50)
            elif gesture == "Kiri":
                tello.move_left(50)
            elif gesture == "Putar_kanan":
                tello.rotate_clockwise(45)
            elif gesture == "Putar_kiri":
                tello.rotate_counter_clockwise(45)
            
            time.sleep(1)
        
        tello.end()
    
    # Menjalankan kontrol drone di thread terpisah
    control_thread = threading.Thread(target=control_tello)
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
