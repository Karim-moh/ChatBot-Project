# modules/image_processing.py

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50

def load_image(image_path):
    """
    Loads an image from the given path and prepares it for feature extraction.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        np.array: Processed image ready for feature extraction.
    """
    img = image.load_img(image_path, target_size=(224, 224))  # Resize to 224x224 as required by ResNet50
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for ResNet50
    return img_array

def extract_image_features(image_path, model):
    """
    Extracts features from an image using a pre-trained ResNet50 model.
    
    Args:
        image_path (str): Path to the image file.
        model (ResNet50): Pre-trained ResNet50 model.
        
    Returns:
        np.array: Extracted feature vector from the image.
    """
    img_array = load_image(image_path)
    features = model.predict(img_array)  # Get features from the model
    return features.flatten()  # Flatten the features into a 1D vector

def load_and_extract_features(sample_folder, model):
    """
    Loads all images from the sample folder and extracts their features.
    
    Args:
        sample_folder (str): Path to the folder containing sample images.
        model (ResNet50): Pre-trained ResNet50 model.
        
    Returns:
        dict: Dictionary where keys are image filenames and values are feature vectors.
    """
    image_features = {}
    for img_name in os.listdir(sample_folder):
        img_path = os.path.join(sample_folder, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            features = extract_image_features(img_path, model)
            image_features[img_name] = features
    return image_features
