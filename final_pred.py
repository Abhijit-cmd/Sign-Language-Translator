# Final_pred.py — Alphabet Fix with Correct Input Shape (224x224x3)

import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import pyttsx3
import shared_state
from web_app import run_flask
from threading import Thread
import enchant
import os

class SignLanguageApp:
    def __init__(self):
        self.labels = self.load_labels('labels.txt')
        self.model = load_model('gesture_alphabet_model.h5')
        self.vs = cv2.VideoCapture(0)
        self.vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.hand_detector = HandDetector(maxHands=1)
        self.speak_engine = self.init_speech_engine()
        self.dictionary = enchant.Dict("en_US")

        self.model_type = "gesture"
        self.conf_threshold = 0.8

        self.gesture_map = {
            "palm": "hello", "fist": "bye", "index": "hi",
            "ok": "ok", "down": "hey", "moved":"beautiful"}

        self.current_symbol = ""
        self.prediction_history = [" "] * 10
        self.current_sentence = ""
        self.suggestions = [" "] * 4
        self.counter = 0
        self.ready_for_next = True

        self.init_gui()
        self.root.after(10, self.video_loop)

    def load_labels(self, path):
        with open(path, 'r') as f:
            return f.read().splitlines()

    def init_speech_engine(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 100)
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[0].id)
        return engine

    def init_gui(self):
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.geometry("1400x750")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=3, width=480, height=640)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=700, y=115, width=400, height=400)

        self.label_char = tk.Label(self.root, text="Character:", font=("Courier", 30, "bold"))
        self.label_char.place(x=10, y=580)
        self.panel_char = tk.Label(self.root, text="", font=("Courier", 30))
        self.panel_char.place(x=280, y=585)

        self.label_sentence = tk.Label(self.root, text="Sentence:", font=("Courier", 30, "bold"))
        self.label_sentence.place(x=10, y=632)
        self.panel_sentence = tk.Label(self.root, text="", font=("Courier", 30))
        self.panel_sentence.place(x=260, y=632)

        self.label_suggestions = tk.Label(self.root, text="Suggestions:", fg="red", font=("Courier", 30, "bold"))
        self.label_suggestions.place(x=10, y=700)

        self.buttons = [tk.Button(self.root, font=("Courier", 20), wraplength=825) for _ in range(4)]
        for i, btn in enumerate(self.buttons):
            btn.place(x=390 + i * 200, y=700)

        self.speak_button = tk.Button(self.root, text="Speak", font=("Courier", 20), command=self.speak_text)
        self.speak_button.place(x=1305, y=630)

        self.clear_button = tk.Button(self.root, text="Clear", font=("Courier", 20), command=self.clear_text)
        self.clear_button.place(x=1205, y=630)

        self.mode_var = tk.StringVar(value=self.model_type)
        self.mode_selector = tk.OptionMenu(self.root, self.mode_var, "gesture", "alphabet", command=self.change_mode)
        self.mode_selector.config(font=("Courier", 16))
        self.mode_selector.place(x=1100, y=20)
        self.mode_label = tk.Label(self.root, text="Mode:", font=("Courier", 20, "bold"))
        self.mode_label.place(x=1000, y=20)

    def change_mode(self, mode):
        self.model_type = mode
        self.clear_text()
        print(f"[INFO] Switched to {mode} mode.")

    def speak_text(self):
        self.speak_engine.say(self.current_sentence)
        self.speak_engine.runAndWait()

    def clear_text(self):
        self.current_sentence = ""
        self.current_symbol = ""
        self.suggestions = [" "] * 4
        self.ready_for_next = True
        self.panel_sentence.config(text="")
        for btn in self.buttons:
            btn.config(text=" ")

    def video_loop(self):
        ret, frame = self.vs.read()
        if not ret:
            self.root.after(10, self.video_loop)
            return

        frame = cv2.flip(frame, 1)
        hands, _ = self.hand_detector.findHands(frame, draw=False)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.display_image(Image.fromarray(frame_rgb), self.panel)

        if hands:
            bbox = hands[0]['bbox']
            x, y, w, h = bbox
            offset = 30
            size = max(w, h) + 2 * offset
            cx, cy = x + w // 2, y + h // 2
            top = max(cy - size // 2, 0)
            bottom = min(cy + size // 2, frame.shape[0])
            left = max(cx - size // 2, 0)
            right = min(cx + size // 2, frame.shape[1])
            cropped = frame[top:bottom, left:right]

            if self.model_type == "alphabet":
                input_img, white_display = self.get_landmark_image(hands[0], left, top, size)
                self.display_image(Image.fromarray(white_display), self.panel2)
                prediction = self.predict_alphabet(input_img)
            else:
                display_img = cv2.resize(cropped, (400, 400))
                self.display_image(Image.fromarray(display_img), self.panel2)
                prediction = self.predict_gesture(cropped)

            if prediction:
                self.handle_prediction(prediction)

        self.update_ui()
        self.root.after(10, self.video_loop)

    def get_landmark_image(self, hand, left, top, size):
        white = np.ones((400, 400, 3), dtype=np.uint8) * 255
        points = []

        for pt in hand['lmList']:
            px = int((pt[0] - left) * (400 / size))
            py = int((pt[1] - top) * (400 / size))
            points.append((px, py))
            cv2.circle(white, (px, py), 2, (0, 255, 0), -1)

        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        ]
        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(white, points[start], points[end], (0, 255, 0), 2)

        input_img = cv2.resize(white, (224, 224)).astype('float32') / 255.0
        input_img = input_img.reshape(1, 224, 224, 3)
        return input_img, white

    def predict_alphabet(self, img):
        try:
            prob = self.model.predict(img, verbose=0)[0]
            index = np.argmax(prob)
            confidence = np.max(prob)
            print(f"[ALPHABET] {self.labels[index]} ({confidence:.2f})")
            return self.labels[index] if confidence > self.conf_threshold else ""
        except Exception as e:
            print(f"[ERROR] Alphabet prediction failed: {e}")
            return ""

    def predict_gesture(self, image):
        try:
            img = cv2.resize(image, (224, 224))
            img = img.astype('float32') / 255.0
            img = img.reshape(1, 224, 224, 3)
            prob = self.model.predict(img, verbose=0)[0]
            index = np.argmax(prob)
            confidence = np.max(prob)
            print(f"[GESTURE] {self.labels[index]} ({confidence:.2f})")
            return self.labels[index] if confidence > self.conf_threshold else ""
        except Exception as e:
            print(f"[ERROR] Gesture prediction failed: {e}")
            return ""

    def handle_prediction(self, pred):
        if not self.ready_for_next or pred == self.current_symbol:
            return

        mapped_word = self.gesture_map.get(pred.lower(), pred)
        if mapped_word.lower() == "space":
            self.current_sentence += " "
        elif mapped_word.lower() == "backspace":
            self.current_sentence = self.current_sentence[:-1]
        elif len(mapped_word) == 1:
            self.current_sentence += mapped_word
        elif mapped_word.lower() in self.gesture_map.values():
            self.current_sentence += f"{mapped_word} "
        else:
            return

        self.current_symbol = pred
        self.update_prediction_history(pred)
        self.update_suggestions()

    def update_prediction_history(self, symbol):
        self.prediction_history[self.counter % 10] = symbol
        self.counter += 1
        shared_state.current_prediction = symbol

    def update_suggestions(self):
        last_word = self.current_sentence.strip().split(" ")[-1]
        if last_word:
            suggestions = self.dictionary.suggest(last_word)
            self.suggestions = suggestions[:4] + [" "] * (4 - len(suggestions))
        else:
            self.suggestions = [" "] * 4

    def display_image(self, img, panel):
        imgtk = ImageTk.PhotoImage(image=img)
        panel.imgtk = imgtk
        panel.config(image=imgtk)

    def update_ui(self):
        self.panel_char.config(text=self.current_symbol)
        self.panel_sentence.config(text=self.current_sentence)
        for i, word in enumerate(self.suggestions):
            self.buttons[i].config(text=word, command=lambda w=word: self.apply_suggestion(w))

    def apply_suggestion(self, word):
        if word.strip():
            words = self.current_sentence.strip().split(" ")
            words[-1] = word
            self.current_sentence = " ".join(words)

    def destructor(self):
        print("Shutting down application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    @property
    def display_sentence(self):
         return self.current_sentence


if __name__ == '__main__':
    print("Starting Application...")
    app = SignLanguageApp()
    Thread(target=run_flask, args=(app,), daemon=True).start()
    app.root.mainloop()
