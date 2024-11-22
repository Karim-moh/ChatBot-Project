from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Initialize NLP models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
