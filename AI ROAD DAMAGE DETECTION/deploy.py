deploy .py
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\ADMIN\Desktop\NexGenMavericks\Czech\train_model.h5')

# Define labels for road damage types
LABELS = ["DeepPothole", "Pothole", "Crack", "AlligatorCrack", "SlightDamage"]

# Function to preprocess the frame for model input
def preprocess_frame(frame):
    # Resize frame to match model input size
    resized_frame = cv2.resize(frame, (224, 224))
    # Normalize pixel values to [0, 1]
    normalized_frame = resized_frame / 255.0
    # Expand dimensions to match model input shape
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
    return preprocessed_frame

# Function to display alerts on frame
def display_alerts(frame, predictions):
    # Iterate through predictions and overlay alerts on frame
    for i, pred in enumerate(predictions):
        if pred > 0.5:  # Threshold for considering a prediction
            label = LABELS[i]
            cv2.putText(frame, label, (50, 50 + 50*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

# Initialize video capture
cap = cv2.VideoCapture(r'C:\Users\ADMIN\Desktop\NexGenMavericks\WhatsApp Video 2024-03-09 at 9.24.12 PM.mp4')

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for model input
    preprocessed_frame = preprocess_frame(frame)

    # Pass frame through the model to get predictions
    predictions = model.predict(preprocessed_frame)[0]

    # Display alerts on the frame based on predictions
    frame_with_alerts = display_alerts(frame, predictions)

    # Display the frame with alerts
    cv2.imshow('Road Damage Detection', frame_with_alerts)

    # Check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
