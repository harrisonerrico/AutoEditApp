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

st.subheader("2. Raw Media Clips Input")
media_input_method = st.radio(
    "Choose input method for raw media:",
    ["Upload ZIP from device", "Provide Google Drive ZIP link", "Provide Google Drive folder link"]
)
media_zip_path = None
media_folder_path = None

if media_input_method == "Upload ZIP from device":
    media_file = st.file_uploader("Upload Raw Media Clips (.zip)", type=["zip"])
    if media_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
            temp_zip.write(media_file.read())
            media_zip_path = temp_zip.name

elif media_input_method == "Provide Google Drive ZIP link":
    media_url = st.text_input("Paste Google Drive direct download link for ZIP file")
    if media_url:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
            response = requests.get(media_url, stream=True)
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    temp_zip.write(chunk)
            media_zip_path = temp_zip.name
        st.success("Media ZIP downloaded successfully.")

elif media_input_method == "Provide Google Drive folder link":
    folder_url = st.text_input("Paste Google Drive folder share link")
    if folder_url:
        def scrape_drive_folder(share_url, dest_folder):
            response = requests.get(share_url)
            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all("a"):
                href = a_tag.get("href", "")
                if "uc?id=" in href or "file/d/" in href:
                    file_id = None
                    if "uc?id=" in href:
                        file_id = href.split("uc?id=")[-1].split("&")[0]
                    elif "file/d/" in href:
                        file_id = href.split("file/d/")[-1].split("/")[0]
                    if file_id:
                        dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                        file_resp = requests.get(dl_url, stream=True)
                        ctype = file_resp.headers.get("Content-Type", "")
                        ext = None
                        if "video/mp4" in ctype:
                            ext = "mp4"
                        elif "video/quicktime" in ctype:
                            ext = "mov"
                        elif "audio/wav" in ctype:
                            ext = "wav"
                        elif "audio/mpeg" in ctype:
                            ext = "mp3"
                        elif "audio/mp4" in ctype or "audio/m4a" in ctype:
                            ext = "m4a"
                        elif "audio/aiff" in ctype:
                            ext = "aiff"
                        elif "audio/aac" in ctype:
                            ext = "aac"
                        if ext:
                            out_path = os.path.join(dest_folder, f"file_{file_id}.{ext}")
                            with open(out_path, "wb") as out_file:
                                for chunk in file_resp.iter_content(32768):
                                    if chunk:
                                        out_file.write(chunk)

        media_folder_path = tempfile.mkdtemp()
        scrape_drive_folder(folder_url, media_folder_path)
        st.success("All supported media files downloaded from folder.")
