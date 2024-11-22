Criminal Feature Matching System
================================

This project involves matching a criminal's description provided as text with images from a dataset. The dataset used is the Illinois DOC labeled faces dataset, and the project involves feature extraction from both text and images to identify the closest matching image based on a given description.

Project Overview
----------------
This project is designed to match descriptions of criminals with a set of images by analyzing facial features. We use a deep learning model for image feature extraction (ResNet50) and a text-based model (Sentence Transformer) for understanding descriptions. The final goal is to display the images that best match the description entered by the user based on various features like age, race, gender, hair color, eye color, etc.

### Features:
- **Image Feature Extraction**: Uses a pretrained ResNet50 model to extract features from the images.
- **Text Feature Extraction**: Uses a zero-shot classification pipeline (`facebook/bart-large-mnli`) to extract textual features and compares them with image features using cosine similarity.
- **Matching System**: Compares the text-based features to images in a database and returns the top 3 most similar images based on similarity scores.
- **UI with Gradio**: The project provides an interface using Gradio where users can input their description and see matching images along with confidence scores.

### Datasets Used:
- **Illinois DOC Labeled Faces Dataset**: Contains images and data related to criminal offenders. This project uses the front face images folder for matching criminal descriptions with photos. A CSV file provides additional metadata (e.g., age, race, gender).

How it Works
------------
1. **Text Feature Extraction**:
   - When a user enters a description, the system processes the text and extracts key attributes like gender, race, hair color, eye color, age group, etc., using a zero-shot classification model.

2. **Image Feature Extraction**:
   - The system preprocesses and extracts features from the provided image dataset using ResNet50.

3. **Matching**:
   - The text features and image features are converted into embeddings (numerical representations), and cosine similarity is used to find the closest match.

4. **Display**:
   - The top 3 most similar images are displayed to the user with similarity scores.

Installation
------------
To install the required packages for this project, run the following command:

    pip install torch torchvision transformers sentence-transformers pandas pillow numpy scikit-learn gradio

Project Structure
-----------------
The project files and folders are organized as follows:

    person_search_project/
│
├── main.py                 # Entry point of the application
├── modules/
│   ├── __init__.py         # Makes the `modules` directory a Python package
│   ├── feature_extraction.py  # Functions for feature extraction
│   ├── preprocessing.py    # Text cleaning, mapping, and preprocessing
│   ├── matching.py         # Matching logic for text and image features
│   ├── display.py          # Functions for displaying images and results
│   ├── nlp_models.py       # Initialization of NLP models
│   └── image_processing.py # Image loading and preprocessing
├── data/
│   ├── person.csv          # Input CSV file containing metadata
│   └── samples/            # Folder containing sample images
└── requirements.txt        # List of required libraries


Usage
-----
1. **Prepare the Data**:
   - Place the sample images in the `samples/` folder.
   - Update `person.csv` with relevant metadata (e.g., ID, race, gender, etc.).

2. **Run the Application**:
   - Execute the following command to start the project:
     
         python main.py

   - This will launch a Gradio interface where you can input descriptions and view matched images.

Documentation
-------------
For detailed information about the project's architecture, usage, and development, refer to the `documentation.docs` file.
