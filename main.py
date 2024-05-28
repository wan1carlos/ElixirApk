import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.utils import platform
from kivy.core.window import Window
from kivy.lang import Builder
import threading
import google.generativeai as genai

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
genai.configure(api_key="AIzaSyAm9XPPs0jmT53_vkBdBFroZzzXgtvckKA")

Builder.load_file('main.kv')

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 0,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

chat_session = model.start_chat()

class FaceCapture(threading.Thread):
    def __init__(self, video_source=0):
        threading.Thread.__init__(self)
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        self.running = False
        self.frame = None
        self.face_frame = None
        self.face_coords = None
        self.avg_rgb = None

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Process frame without rotation
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.face_frame, self.face_coords, self.avg_rgb = self.detect_face(self.frame.copy())
                self.check_lighting()
                self.check_face_detected()
            else:
                break
        self.cap.release()

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            padding = 20
            x -= padding
            y -= padding
            w += 2 * padding
            h += 2 * padding
            face_frame = frame[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)]
            avg_rgb = np.mean(face_frame, axis=(0, 1))
            text_position = (10, 20)
            font_scale = 0.8  # Adjust the font size here
            cv2.putText(frame, f"RGB: {avg_rgb}", text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
            return face_frame, (x, y, w, h), avg_rgb
        return None, None, None

    def check_lighting(self):
        brightness = np.mean(cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY))
        text_position = (10, 50)
        font_scale = 0.8  # Adjust the font size here
        if brightness < 50:
            text = "Warning: Lighting is too dim!"
            cv2.putText(self.frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA)
        elif brightness > 200:
            text = "Warning: Lighting is too bright!"
            cv2.putText(self.frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA)

    def check_face_detected(self):
        text_position = (10, 80)
        font_scale = 0.8  # Adjust the font size here
        if self.face_frame is None:
            text = "Warning: No face detected!"
            cv2.putText(self.frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA)

    def stop(self):
        self.running = False

class FaceColorDetectorApp(App):
    def build(self):
        self.capture_thread = FaceCapture()
        self.capture_thread.start()

        layout = BoxLayout(orientation='vertical')

        self.image_widget = Image(size_hint=(1, 0.8))
        layout.add_widget(self.image_widget)

        button_layout = BoxLayout(size_hint=(1, 0.2))

        capture_button = Button(text="Capture Face", on_press=self.capture_face)
        button_layout.add_widget(capture_button)

        quit_button = Button(text="Quit", on_press=self.quit_app)
        button_layout.add_widget(quit_button)

        layout.add_widget(button_layout)

        Clock.schedule_interval(self.update_video, 1.0 / 30.0)
        return layout

    def update_video(self, dt):
        if hasattr(self.capture_thread, 'frame'):
            frame = self.capture_thread.frame
            # Rotate frame for display only
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.flip(frame, 1)
            buf = frame.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            image_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.image_widget.texture = image_texture

    def capture_face(self, instance):
        if hasattr(self.capture_thread, 'face_frame') and hasattr(self.capture_thread, 'face_coords') and hasattr(self.capture_thread, 'avg_rgb'):
            # Print the RGB values inside the frame of the captured face
            avg_rgb_str = f"Average RGB Color: {self.capture_thread.avg_rgb}"
            face_frame_bgr = cv2.cvtColor(self.capture_thread.face_frame, cv2.COLOR_RGB2BGR)
            font_scale = 0.45  # Adjust the font size here
            font_face = cv2.FONT_HERSHEY_SIMPLEX  # Change the font style here
            font_thickness = 2  # Adjust the thickness of the text here
            cv2.putText(face_frame_bgr, avg_rgb_str, (10, 30), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            cv2.imshow('Captured Face', face_frame_bgr)
            print("Average RGB Color of Face:", self.capture_thread.avg_rgb)

            # Send the RGB value to the chat session and get the response text
            prompt_values = str(self.capture_thread.avg_rgb)
            pre_prompt = "Recommend the best color palette and do a seasonal color analysis for these rgb color values from the face "
            response = chat_session.send_message(pre_prompt + prompt_values)

            # Print and display the response text
            print(response.text)
            self.display_response(response.text)

    def display_response(self, response_text):
        # Create a black image with appropriate size
        response_image = np.zeros((1000, 800, 3), np.uint8)

        # Define font properties
        font_scale = 0.5  # Adjust the font size here
        font_face = cv2.FONT_HERSHEY_SIMPLEX  # Change the font style here
        font_thickness = 1  # Adjust the thickness of the text here

        # Split the response text into lines
        lines = response_text.split('\n')
        y = 30  # Initial y-coordinate for the first line

        # Draw each line of the response text on the image
        for line in lines:
            cv2.putText(response_image, line, (10, y), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            y += 30  # Move to the next line

        # Display the response text image in a window
        cv2.namedWindow('Response', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Response', 1000, 800)  # Resize window for better text display
        cv2.imshow('Response', response_image)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed to close the window

    def quit_app(self, instance):
        self.capture_thread.stop()
        App.get_running_app().stop()

if __name__ == '__main__':
    FaceColorDetectorApp().run()
    __version__ = "0.1"
