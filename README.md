## Road Damage Detection Using Deep Learning
**Project Overview**

This project implements a deep learning-based system for detecting and classifying road damage. The model uses a fine-tuned ResNet50 architecture to identify damage types such as cracks, potholes, ruts, and spalls. It supports both real-time video streams and static image inputs, making it versatile for road condition monitoring and infrastructure maintenance planning.

# How It Works

**Data Preparation:**

Training data includes images of roads with XML annotations specifying damage locations and types.
Images are preprocessed and resized to match ResNet50 input dimensions (224x224).

**Model Training:**

The ResNet50 model is fine-tuned with additional dense layers for damage classification.
Training uses categorical cross-entropy loss, with callbacks for model checkpointing and performance optimization.

**Deployment:**

The system processes video frames or uploaded images, detects damage, and overlays bounding boxes with labels on detected areas.
OpenCV is used for input handling, visualization, and real-time feedback.
# Technical Description

**Model Architecture:**
A pre-trained ResNet50 serves as the backbone for feature extraction. Custom dense layers are added to classify damage into specific categories.

**Input Handling:**

Images: Loaded, preprocessed, and passed to the model for predictions.

Videos: Frames are extracted, preprocessed, and evaluated in real-time.

Outputs:Bounding boxes highlight damage areas, with labels specifying the damage type or indicating no damage.

**Training Details:**

Batch size: 8

Epochs: 50

Optimizer: Adam

Loss function: Categorical Cross-Entropy

**Required Libraries**

Install the following Python libraries before running the project:

```bash
pip install tensorflow keras opencv-python numpy
```

