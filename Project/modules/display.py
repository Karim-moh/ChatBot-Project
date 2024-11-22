import matplotlib.pyplot as plt
from PIL import Image

def display_images(top_matches):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Top 3 Matched Images", fontsize=16)

    for i, match in enumerate(top_matches):
        img = Image.open(match['image_path'])
        axes[i].imshow(img)
        axes[i].set_title(f"Match {i+1}\nSimilarity: {match['similarity']:.2f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
