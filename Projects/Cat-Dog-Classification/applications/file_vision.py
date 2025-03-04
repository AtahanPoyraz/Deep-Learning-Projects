import os
import cv2
import numpy as np

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from keras.api.models import load_model

class Model:
    def __init__(self, path) -> None:
        self.path = path
    
    def build(self):
        self.model = load_model(self.path)

    def predict(self, path : str):
        X_new = cv2.imread(path, cv2.IMREAD_COLOR)
        X_new = cv2.resize(X_new, (64, 64))
        X_new = np.array(X_new).reshape(-1, 64, 64, 3) / 255.0

        y_proba = self.model.predict(X_new)
        y_pred = (y_proba > 0.5).astype("int32")
        preds = np.array(["Dog", "Cat"])[y_pred]

        return preds
    
class FileVision(Model):
    def __init__(self, path : str) -> None:
        super(FileVision, self).__init__(path)
        self.build()

    def initialize_ui(self):
        self.root = tk.Tk()
        self.root.title("Classificator")
        self.root.geometry("350x400")
        self.root.resizable(False, False)

        self.button_frame = tk.LabelFrame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.predict_frame = tk.Frame(self.root)
        self.predict_frame.pack(side=tk.BOTTOM, padx=0, pady=10)

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.open_button = tk.Button(self.button_frame, text="Open", command=self.open_file, width=12)
        self.open_button.grid(padx=5, pady=5, row=1, column=1)

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.model_predict, width=12)
        self.predict_button.grid(padx=5, pady=5, row=1, column=2)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_display, width=12)
        self.clear_button.grid(padx=5, pady=5, row=1, column=3)

        self.predict_box = tk.Text(self.predict_frame, width=16, height=1)
        self.predict_box.pack()

        self.img_label = tk.Label(self.image_frame)
        self.img_label.pack()
        self.root.mainloop()

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.test_path = file_path
            self.show_image(file_path)

        else:
            print("File do not selected.")

    def model_predict(self):
        result = self.predict(self.test_path)
        self.predict_box.insert("1.0", f"{' ' * 6}{result[0][0]}")

    def show_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((375, 275), Image.FIXED)
        photo = ImageTk.PhotoImage(img)

        self.img_label.configure(image=photo)
        self.img_label.image = photo

    def clear_display(self):
        self.img_label.configure(image=None)
        self.img_label.image = None
        self.predict_box.delete("1.0", "end")

if __name__ == "__main__":
    FileVision("Projects/Cat-Dog-Classification/artifacts/model.h5").initialize_ui()