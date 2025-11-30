# nlp_model.py
from transformers import pipeline
import joblib

# Load your candidate labels (from your .pkl if you saved them)
try:
    candidate_labels = joblib.load("candidate_labels.pkl")
except:
    candidate_labels = [
        "a request for a donation",
        "a demand for a ransom payment",
        "a piece of security advice",
        "an advertisement for a product or service",
        "a warning about a scam",
        "a discussion about cybercrime tools or software",
        "an advertisement for stolen financial information",
        "a scam pretending to be a marketplace or service",
        "a warning or report about law enforcement activity",
        "a neutral or unrelated type of content"
    ]

print("Loading zero-shot classifier… this may take 20-40 seconds")
classifier = pipeline(
    "zero-shot-classification",
    model="roberta-large-mnli"
)
print("Zero-shot classifier loaded.")
