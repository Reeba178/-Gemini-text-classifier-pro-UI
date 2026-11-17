# Gemini Studio — Text Intelligence

A refreshed desktop interface for text classification and sentiment analysis with Google's Gemini API.

## Files
- `app.py` — application
- `requirements.txt` — required Python packages
- `.env` — your local Gemini API key and default model
- `.env.example` — safe configuration template
- `.gitignore` — excludes the local `.env` file from Git

## Setup (Windows / Python 3.15)
1. Extract this ZIP.
2. Open `.env` and replace `PASTE_YOUR_GEMINI_API_KEY_HERE` with your key.
3. Open Command Prompt in this folder and run:

   ```bash
   py -3.15 -m pip install -r requirements.txt
   py -3.15 app.py
   ```

If `py -3.15` is not recognized, use `python` if it points to your Python 3.15 installation.

## Features
- Modern light interface with responsive scrolling layout
- Topic/category classification
- Positive / Negative / Neutral sentiment
- Sentiment score and confidence
- Emotion, keywords, summary, and explanation
- Optional custom categories
- Copy result and export analysis history to CSV

## Privacy and API key
Text submitted for analysis is sent to Google's Gemini API. Keep `.env` private and never share your API key. If a key is exposed, revoke it and create a new one.

## Model note
The default model identifier is `gemini-3.1-flash-lite`. If Google AI Studio reports that this model is unavailable to your account, replace `GEMINI_MODEL` in `.env` and restart the application, or enter an available model ID in the model field.
