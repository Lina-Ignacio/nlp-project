import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model
tokenizer = BertTokenizer.from_pretrained("model")
model = BertForSequenceClassification.from_pretrained("model")

st.title("Sentiment Analysis App")
st.write("Enter a sentence to analyze its sentiment.")

text = st.text_area("Text input")

if st.button("Analyze"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits).item()

    if prediction == 1:
        st.success("😊 Positive sentiment")
    else:
        st.error("😠 Negative sentiment")
