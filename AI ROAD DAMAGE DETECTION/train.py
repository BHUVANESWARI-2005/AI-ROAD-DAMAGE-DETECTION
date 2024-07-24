
import os
import numpy as np
import tensorflow as tf
import keras

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import xml.etree.ElementTree as ET

# Define constants
LABELS = ["D00", "D10", "D20", "D40"]
IMAGE_SIZE = (224, 224)  # Image size for ResNet50 input
BATCH_SIZE = 8
EPOCHS = 50


# Load and preprocess data
def load_images_and_labels(image_paths, annotation_paths):
    images = []
    labels = []
    for img_path, ann_path in zip(image_paths, annotation_paths):
        image = load_img(img_path, target_size=IMAGE_SIZE)
        image = img_to_array(image)
        images.append(image)

        # Load and process annotations to extract class labels
        annotation_labels = load_annotation_function(ann_path)
        labels.append(annotation_labels)

    return np.array(images), labels


def load_annotation_function(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Extract class labels (road damage types) from annotation
    labels = []
    if root.find('object') is not None:
        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(label)
    else:
        labels.append("NoDamage")  # Placeholder label for images without damage

    return labels


# Define data directory for the selected region
region_train_data_dir = r"C:\Users\ADMIN\Desktop\NexGenMavericks\Czech\train"

# Combine paths for images and annotations
image_dir = os.path.join(region_train_data_dir, "images")
annotation_dir = os.path.join(region_train_data_dir, "annotations", "xmls")

image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]
annotation_paths = [os.path.join(annotation_dir, ann) for ann in os.listdir(annotation_dir) if ann.endswith('.xml')]

# Load images and labels
train_images, train_labels = load_images_and_labels(image_paths, annotation_paths)

# Preprocess images
train_images = train_images / 255.0

# Convert labels to categorical format
train_labels = [[LABELS.index(label) if label in LABELS else len(LABELS) for label in labels] for labels in
                train_labels]
train_labels = [to_categorical(labels, num_classes=len(LABELS) + 1) for labels in train_labels]

# Flatten the list of labels
train_labels_flat = [item for sublist in train_labels for item in sublist]

# Define ResNet50 model as backbone
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(LABELS) + 1, activation='softmax')(x)  # Add one more class for images without damage

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)

# Train the model
model.fit(x=train_images, y=np.array(train_labels_flat), validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS,
          callbacks=[checkpoint])

# Save trained model
model.save("trained_model.h5")
