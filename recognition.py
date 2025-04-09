import cv2
import numpy as np
import json
import requests
import threading
from kivy.app import App
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock, mainthread
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.metrics import dp
from tensorflow.keras.models import load_model

# Load model and class indices
model = load_model('fruit_recognition_model.h5')
with open('class_indices.json', 'r') as f:
    fruit_classes = json.load(f)

def get_recipes(fruit_name):
    api_key = "7854e31eab6b446cae9ffda9c0fe1c67"
    url = "https://api.spoonacular.com/recipes/complexSearch"
    params = {"query": fruit_name, "apiKey": api_key, "number": 5}
    response = requests.get(url, params=params)
    if response.status_code == 402:
        return ["API limit reached. Try again later."]
    elif response.status_code == 200:
        recipes = response.json().get("results", [])
        return [recipe["title"] for recipe in recipes] if recipes else ["No recipes found."]
    return ["Error fetching recipes."]

def get_nutrition(fruit_name):
    api_key = "AruAFSTx9fcRdxMaHf5I9p696DotbfO8W1v2HWYp"
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"query": fruit_name, "api_key": api_key, "pageSize": 1}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        foods = data.get("foods", [])
        if foods:
            nutrients = foods[0].get("foodNutrients", [])
            return {nutrient["nutrientName"]: nutrient["value"] for nutrient in nutrients[:5]}
    return {"error": "No data found."}

class FruitRecognitionApp(App):
    def build(self):
        Window.clearcolor = (0, 0, 0, 1)  # Dark background

        # Main layout
        self.layout = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))

        # Top label: Name & Confidence
        self.result_label = Label(
            text="Name & Confidence",
            font_size='18sp',
            size_hint=(1, 0.08),
            halign='center',
            valign='middle',
            color=(1, 1, 1, 1)
        )
        self.layout.add_widget(self.result_label)

        # Camera preview
        self.image = Image(size_hint=(1, 0.6))
        self.layout.add_widget(self.image)

        # Stop/Start button - now starts as "Start"
        self.stop_button = Button(
        text="Start",
        size_hint=(1, 0.08),
        background_color=(0.2, 0.8, 0.2, 1),  # Green for Start
        background_normal='',
        font_size='18sp'
    )
        self.stop_button.bind(on_press=self.toggle_camera)
        self.layout.add_widget(self.stop_button)

        # Bottom info layout (Nutrition & Recipes)
        bottom_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.24), padding=[dp(10), 0], spacing=dp(10))

        # Nutrition Info Label
        self.nutrition_title = Label(
            text="Nutrition Info",
            font_size='18sp',
            halign='left',
            valign='top',
            color=(1, 1, 1, 1)
        )
        bottom_layout.add_widget(self.nutrition_title)

        # Recipes Label
        self.recipes_label = Label(
            text="Recipes",
            font_size='18sp',
            halign='right',
            valign='top',
            color=(1, 1, 1, 1)
        )
        bottom_layout.add_widget(self.recipes_label)

        self.layout.add_widget(bottom_layout)

        self.capture = None
        self.running = False
        return self.layout

    def toggle_camera(self, instance):
        if self.running:
            self.running = False
            self.stop_button.text = "Start"
            self.stop_button.background_color = (0.2, 0.8, 0.2, 1)
            if self.capture:
                self.capture.release()
        else:
            self.running = True
            self.stop_button.text = "Stop"
            self.stop_button.background_color = (0.8, 0.2, 0.2, 1)
            self.capture = cv2.VideoCapture(0)
            Clock.schedule_interval(self.update, 1.0 / 15.0)

    def preprocess_image(self, frame):
        resized = cv2.resize(frame, (100, 100))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def predict(self, input_tensor):
        predictions = model.predict(input_tensor)
        return predictions

    def update(self, dt):
        if not self.running or self.capture is None:
            return

        ret, frame = self.capture.read()
        if not ret:
            return

        input_tensor = self.preprocess_image(frame)
        predictions = self.predict(input_tensor)
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        fruit_name = fruit_classes[class_idx]

        self.result_label.text = f"{fruit_name} ({confidence:.2%})"
        threading.Thread(target=self.fetch_info, args=(fruit_name,), daemon=True).start()

        frame = cv2.flip(frame, 0)
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def fetch_info(self, fruit_name):
        if not self.running:
            return

        recipes = get_recipes(fruit_name)
        nutrition = get_nutrition(fruit_name)

        if not self.running:
            return

        recipes_text = "\n".join([f"• {r}" for r in recipes])
        nutrition_text = "\n".join([f"{k}: {v}" for k, v in nutrition.items()])
        self.update_ui(recipes_text, nutrition_text)

    @mainthread
    def update_ui(self, recipes_text, nutrition_text):
        self.recipes_label.text = recipes_text
        self.nutrition_title.text = f"Nutrition Info\n{nutrition_text}"

    def on_stop(self):
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    FruitRecognitionApp().run()
