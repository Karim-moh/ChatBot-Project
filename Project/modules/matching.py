import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from modules.feature_extraction import extract_image_features
from modules.preprocessing import clean_description

def match_description_to_images(description, samples_folder, person_df):
    person_df = filter_person_df_with_images(person_df, samples_folder)

    # Extract text features
    text_features = extract_text_features(description)

    # Filter person_df based on extracted features
    filtered_df = person_df.copy()

    if text_features['race']:
        filtered_df = filtered_df[filtered_df['race'].str.lower() == text_features['race'].lower()]

    if text_features['sex']:
        filtered_df = filtered_df[filtered_df['sex'].str.lower() == text_features['sex'].lower()]

    if text_features['hair']:
        filtered_df = filtered_df[filtered_df['hair'].str.lower() == text_features['hair'].lower()]

    if text_features['eyes']:
        filtered_df = filtered_df[filtered_df['eyes'].str.lower() == text_features['eyes'].lower()]

    # Extract image features for filtered persons
    image_features = {}
    for _, row in filtered_df.iterrows():
        image_path = os.path.join(samples_folder, f"{row['id']}.jpg")
        if os.path.isfile(image_path):
            features = extract_image_features(image_path)
            image_features[row['id']] = features

    if not image_features:
        return []  # No matches found

    # Compare text features with image features
    text_embedding = get_text_embedding(text_features)
    image_embeddings = np.array(list(image_features.values()))
    image_ids = list(image_features.keys())

    top_indices, similarities = compare_embeddings(text_embedding, image_embeddings)

    # Get top 3 matches
    top_matches = []
    for idx, similarity in zip(top_indices[:3], similarities[:3]):
        image_id = image_ids[idx]
        person_info = filtered_df[filtered_df['id'].astype(str) == str(image_id)].iloc[0].to_dict()

        match_info = {
            'image_name': f"{image_id}.jpg",
            'image_path': os.path.join(samples_folder, f"{image_id}.jpg"),
            'similarity': similarity,
            'person_info': person_info
        }
        top_matches.append(match_info)

    return top_matches
