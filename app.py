import os
import subprocess
import tempfile
import numpy as np
import wave
import contextlib
import xml.etree.ElementTree as ET
from datetime import timedelta
from PIL import Image
import torch
import clip
import cv2
import openai
import json
import whisper
import streamlit as st
import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import zipfile
import shutil
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="Smart Auto Edit", layout="wide")
st.title("🎬 Smart Auto Edit – Reference-Based Editor")

def download_large_file_from_drive(file_id, output_path):
    session = requests.Session()
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(download_url, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        download_url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
        response = session.get(download_url, stream=True)

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

st.subheader("1. Reference Video Input")
ref_input_method = st.radio(
    "Choose input method for the reference video:",
    ["Upload from device", "Provide Google Drive link"]
)
reference_path = None
if ref_input_method == "Upload from device":
    reference_file = st.file_uploader("Upload Edited Reference Video", type=["mp4", "mov"])
    if reference_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(reference_file.name)[1]) as temp_ref:
            temp_ref.write(reference_file.read())
            reference_path = temp_ref.name
else:
    reference_url = st.text_input("Paste Google Drive share link for reference video")
    if reference_url:
        match = re.search(r"(?:file/d/|id=)([a-zA-Z0-9_-]{10,})", reference_url)
        if match:
            file_id = match.group(1)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_ref:
                try:
                    download_large_file_from_drive(file_id, temp_ref.name)
                    reference_path = temp_ref.name
                    st.success("Reference video downloaded successfully.")
                except Exception as e:
                    st.error(f"Failed to download reference video. Error: {e}")

# The rest of your code remains unchanged...

st.subheader("2. Raw Media Clips Input")
media_input_method = st.radio(
    "Choose input method for raw media:",
    ["Upload ZIP from device", "Provide Google Drive ZIP link", "Provide Google Drive folder link"]
)
# ... (rest of your existing code follows) ...
