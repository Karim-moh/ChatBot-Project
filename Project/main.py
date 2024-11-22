import os
import pandas as pd
from modules.feature_extraction import extract_text_features, extract_image_features, extract_features_for_samples
from modules.preprocessing import clean_description
from modules.matching import match_description_to_images
from modules.display import display_images
from modules.nlp_models import initialize_nlp_models

def main():
    samples_folder = "data/samples"
    person_df = pd.read_csv("data/person.csv", delimiter=';')

    while True:
        description = input("Enter a description of the person you're looking for (or 'quit' to exit): ")
        if description.lower() == 'quit':
            print("Thank you for using the person search system. Goodbye!")
            break

        top_matches = match_description_to_images(description, samples_folder, person_df)

        if not top_matches:
            print("No matches found for the given description.")
        else:
            print("\nTop 3 matched images:")
            for i, match in enumerate(top_matches, 1):
                print(f"\n{i}. Image: {match['image_name']}")
                print(f"   Similarity Score: {match['similarity']:.2f}")
                print("   Person Information:")
                for key, value in match['person_info'].items():
                    print(f"   - {key.capitalize()}: {value}")

            # Display the images
            display_images(top_matches)

if __name__ == "__main__":
    main()
