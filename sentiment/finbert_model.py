# sentiment/finbert_model.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "ProsusAI/finbert"
labels = ["negative", "neutral", "positive"]

@st.cache_resource  # FIX: loads ONCE, reused forever across all reruns
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()  # set to eval mode — faster inference
    return tokenizer, model

def finbert_predict(text: str):
    tokenizer, model = load_finbert()  # cached — instant after first load

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    scores = {labels[i]: float(probs[i]) for i in range(len(labels))}

    sentiment = max(scores, key=scores.get)
    confidence = scores[sentiment]

    return sentiment, confidence, scores