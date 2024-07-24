test.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load the trained model
model = load_model(r"C:\Users\ADMIN\Desktop\NexGenMavericks\Czech\train_model.h5")

# Define constants for road damage types
LABELS = ["Crack", "Pothole", "Rut", "Spall"]
IMAGE_SIZE = (224, 224)

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Resize image to the input size of the model
    image = cv2.resize(image, IMAGE_SIZE)
    # Convert image to array and preprocess for ResNet50 model
    image = img_to_array(image)
    image = preprocess_input(image)
    return image

# Function to draw bounding boxes around detected damage areas
def draw_damage_boxes(image, predictions):
    for i, prediction in enumerate(predictions):
        # Get the predicted label
        predicted_label_index = np.argmax(prediction)
        if predicted_label_index < len(LABELS):
            predicted_label = LABELS[predicted_label_index]
            # If damage is detected, draw a bounding box around it
            if predicted_label != "NoDamage":
                # Draw a red rectangle around the damaged area
                cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), 2)
            else:
                # Draw a green rectangle around the undamaged area
                cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
            cv2.putText(image, f"Damage: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

# Open file dialog to select an image
Tk().withdraw()  # Hide the main tkinter window
image_path = askopenfilename(title="Select an image file", filetypes=[("Image files", ".jpg;.jpeg;*.png")])

# Preprocess the uploaded image
image = preprocess_image(image_path)

# Make predictions using the model
predictions = model.predict(np.expand_dims(image, axis=0))

# Draw bounding boxes around detected damage areas
processed_image = draw_damage_boxes(image.copy(), predictions)

# Display the processed image
cv2.imshow('Damage Detection', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
