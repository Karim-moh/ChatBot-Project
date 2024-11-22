from transformers import pipeline
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)

# Load pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize NLP models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def extract_image_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet50(img_tensor).squeeze().cpu().numpy()
    return features

def extract_text_features(text):
    feature_list = [
        'male', 'female', 'young', 'middle-aged', 'elderly',
        'blonde hair', 'brown hair', 'black hair', 'grey hair', 'white hair', 'red hair',
        'blue eyes', 'green eyes', 'brown eyes', 'hazel eyes',
        'bald', 'white', 'black', 'asian', 'hispanic', 'indian',
        'tall', 'short', 'overweight', 'slim'
    ]

    logging.info(f"Processing text: '{text}'")
    results = classifier(text, feature_list, multi_label=True)

    confidence_threshold = 0.7
    attributes = {
        'sex': None,
        'age': None,
        'hair': None,
        'eyes': None,
        'height': None,
        'race': None,
        'weight': None
    }

    temp_attributes = {
        'hair': [],
        'age': [],
        'height': []
    }

    for label, score in zip(results['labels'], results['scores']):
        if score > confidence_threshold:
            if label in ['male', 'female']:
                attributes['sex'] = label
            elif label in ['young', 'middle-aged', 'elderly']:
                temp_attributes['age'].append((label, score))
            elif 'hair' in label:
                temp_attributes['hair'].append((label.split()[0], score))
            elif 'eyes' in label:
                attributes['eyes'] = label.split()[0]
            elif label in ['tall', 'short']:
                temp_attributes['height'].append((label, score))
            elif label in ['overweight', 'slim']:
                attributes['weight'] = label
            elif label in ['white', 'black', 'asian', 'hispanic', 'indian']:
                attributes['race'] = label

    if temp_attributes['hair']:
        attributes['hair'] = max(temp_attributes['hair'], key=lambda x: x[1])[0]
    if temp_attributes['age']:
        attributes['age'] = max(temp_attributes['age'], key=lambda x: x[1])[0]
    if temp_attributes['height']:
        attributes['height'] = max(temp_attributes['height'], key=lambda x: x[1])[0]

    logging.info(f"Extracted features: {attributes}")
    return attributes
