import streamlit as st
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("toxic_distilbert/toxic_distilbert")
    model = DistilBertForSequenceClassification.from_pretrained("toxic_distilbert/toxic_distilbert")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

labels = [
    "Toxic",
    "Severe Toxic",
    "Obscene",
    "Threat",
    "Insult",
    "Identity Hate"
]

def predict(text):
    inputs = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).numpy()[0]

    return probs

st.set_page_config(page_title="YouTube Toxic Comment Detector", layout="centered")
st.title("YouTube Toxic Comment Detection")

comment = st.text_area("Enter a YouTube comment:", height=150)

if st.button("Analyze"):
    if comment.strip() == "":
        st.warning("Please enter a comment.")
    else:
        probs = predict(comment)
        st.subheader("Prediction Results")

        for label, score in zip(labels, probs):
            st.write(f"**{label}:** {score*100:.2f}%")

