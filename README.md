# Smart Auto Edit – Reference-Based Editor

A Streamlit app to automatically edit video using a reference clip, matching visual style, motion, and audio.

## Features
- Scene detection from reference video
- Whisper-based voice/music transcription
- GPT-4 powered shot matching
- Visual similarity via OpenAI CLIP
- Audio sync and CapCut XML export

## Run Locally
```bash
pip install -r requirements.txt
streamlit run AutoEditApp.py
```

## Deployment
Push to GitHub and deploy via [Streamlit Cloud](https://streamlit.io/cloud).
