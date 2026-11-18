import streamlit as st
import google.generativeai as genai
import os

st.set_page_config(page_title="Gemini Studio - Text Intelligence", layout="wide")
st.title("✨ Gemini Studio — Text Intelligence")
st.write("AI-powered text classification & sentiment analysis")

# API Key
api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
if not api_key:
    api_key = os.getenv("GEMINI_API_KEY", "")

if not api_key:
    st.warning("Please add your GEMINI_API_KEY in Streamlit Secrets to use this app.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("model = genai.GenerativeModel("gemini-3.1-flash-lite")

text = st.text_area("Enter text to analyze:", height=150, placeholder="Type or paste your text here...")

col1, col2 = st.columns(2)
with col1:
    classify = st.button("🔍 Classify Text", use_container_width=True)
with col2:
    sentiment = st.button("💭 Analyze Sentiment", use_container_width=True)

if classify and text:
    with st.spinner("Classifying..."):
        res = model.generate_content(f"Classify this text into a category (e.g. News, Sports, Technology, Business, Entertainment, etc.) and explain why in 1 line. Text: {text}")
        st.success("Classification Result")
        st.write(res.text)

if sentiment and text:
    with st.spinner("Analyzing sentiment..."):
        res = model.generate_content(f"Analyze the sentiment of this text (Positive / Negative / Neutral) with confidence score and 1 line reason. Text: {text}")
        st.success("Sentiment Result")
        st.write(res.text)

if (classify or sentiment) and not text:
    st.error("Please enter some text first!")
