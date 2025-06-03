import os
import subprocess
import tempfile
import numpy as np
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

# ----------------------
# 1. Reference Video Input
# ----------------------
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
    def download_drive_file(file_url, output_path):
        """
        Download a Google Drive file (handles large-file confirmation).
        Returns (True, debug_info) if successful, (False, debug_info) otherwise.
        debug_info is a dict: {'content_type': str, 'bytes_written': int}
        """
        match = re.search(r"(?:file/d/|id=)([a-zA-Z0-9_-]{10,})", file_url)
        if not match:
            return False, {"error": "Could not find file ID in URL."}
        file_id = match.group(1)

        session = requests.Session()
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = session.get(download_url, stream=True)

        # If Drive sets a "download_warning" cookie, we need to confirm
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        if token:
            download_url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
            response = session.get(download_url, stream=True)

        content_type = response.headers.get("Content-Type", "unknown")
        # If it's HTML, we're not getting raw video bytes
        if "text/html" in content_type:
            return False, {"content_type": content_type}

        bytes_written = 0
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)

        return True, {"content_type": content_type, "bytes_written": bytes_written}

    reference_url = st.text_input("Paste Google Drive share link for reference video")
    if reference_url:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_ref:
            success, info = download_drive_file(reference_url, temp_ref.name)
            if success:
                reference_path = temp_ref.name
                st.success("Reference video downloaded successfully.")
            else:
                st.error(f"Failed to download a valid video. Content-Type: {info.get('content_type')}, Error: {info.get('error', '')}")
                st.info("You can switch to 'Upload from device' to manually upload the file.")

# ----------------------
# 2. Raw Media Clips Input
# ----------------------
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
    media_url = st.text_input("Paste Google Drive share link for ZIP file")
    if media_url:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
            response = requests.get(media_url, stream=True)
            # For large ZIPs, you might need the same token logic as above—assuming small enough for now
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

# ----------------------
# 3. Processing Logic
# ----------------------
scenes = []
edit_guide = []
shot_data = []
transcription_data = []

@st.cache_data
def transcribe_and_classify_audio(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    segments = result.get("segments", [])
    transcription = []
    for segment in segments:
        text = segment.get("text", "")
        if any(keyword in text.lower() for keyword in ["uh", "um", "like", "i think", "you know"]):
            segment_type = "speech"
        elif any(char.isalpha() for char in text) and text.strip().endswith("."):
            segment_type = "speech"
        else:
            segment_type = "music"
        transcription.append({
            "start": segment["start"],
            "end": segment["end"],
            "type": segment_type,
            "text": text
        })
    return transcription

@st.cache_data
def extract_middle_frame(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame = cap.read()
    cap.release()
    return frame if success else None

def extract_frame_embedding(frame, model, preprocess):
    if frame is None:
        return torch.zeros((512,))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)
    image_input = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image_input)
    return features[0].numpy()

def estimate_motion(video_path, start, end):
    return abs(end - start)

def get_gpt_edit_guide(features):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    messages = [
        {
            "role": "system", 
            "content": "You are a professional video editor assistant. Given shot-level features, describe their purpose and how they contribute to the edit."
        },
        {
            "role": "user", 
            "content": f"Here is a list of shots with features: {json.dumps(features)}. Provide a JSON list of guidance on how to recreate these shots using different clips based on similarity of motion, brightness, and duration."
        }
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.5
        )
        return json.loads(response["choices"][0]["message"]["content"])
    except:
        return []

def analyze_reference(reference_path):
    video_manager = VideoManager([str(reference_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

def analyze_shot_structure(reference_path, scenes, model, preprocess):
    results = []
    for start, end in scenes:
        duration = end - start
        frame = extract_middle_frame(reference_path, start + duration / 2)
        motion = estimate_motion(reference_path, start, end)
        embedding = extract_frame_embedding(frame, model, preprocess)
        results.append({
            "start": start,
            "end": end,
            "duration": duration,
            "motion": motion,
            "embedding": embedding.tolist()
        })
    return results

def assign_raw_clips(media_folder, shot_data, model, preprocess):
    valid_extensions = (".mp4", ".mov", ".wav", ".mp3", ".m4a", ".aiff", ".aac")
    for shot in shot_data:
        best_score = float("-inf")
        best_path = None
        for root, _, files in os.walk(media_folder):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    path = os.path.join(root, file)
                    if file.lower().endswith((".mp4", ".mov")):
                        frame = extract_middle_frame(path, 1.0)
                        embedding = extract_frame_embedding(frame, model, preprocess)
                        score = np.dot(shot["embedding"], embedding) / (
                            np.linalg.norm(shot["embedding"]) * np.linalg.norm(embedding) + 1e-6
                        )
                        if score > best_score:
                            best_score = score
                            best_path = path
        shot["match"] = best_path
    return shot_data

def generate_auto_edit(shot_data, transcription_data):
    audio_overlays = [seg for seg in transcription_data if seg["type"] == "speech"]
    music_overlays = [seg for seg in transcription_data if seg["type"] == "music"]

    for i, shot in enumerate(shot_data):
        output_name = f"output_clip_{i+1:03d}.mp4"
        if shot.get("match"):
            start_time = 0
            cmd = [
                "ffmpeg", "-y", "-i", shot["match"],
                "-ss", str(start_time), "-t", str(shot["duration"]),
                "-vf", "scale=1920:1080", output_name
            ]
            subprocess.run(cmd)

    if audio_overlays:
        with open("audio_overlay.txt", "w") as f:
            for seg in audio_overlays:
                f.write(f"Speech segment: {seg['start']}s to {seg['end']}s\n")
    if music_overlays:
        with open("music_overlay.txt", "w") as f:
            for seg in music_overlays:
                f.write(f"Music segment: {seg['start']}s to {seg['end']}s\n")

def export_to_capcut_xml(shot_data):
    root = ET.Element("project")
    timeline = ET.SubElement(root, "timeline")
    for shot in shot_data:
        if shot.get("match"):
            clip_elem = ET.SubElement(timeline, "clip", {
                "src": shot["match"],
                "start": str(0),
                "duration": str(int(shot["duration"] * 1000))
            })
    tree = ET.ElementTree(root)
    tree.write("capcut_export.xml")

if reference_path and (media_zip_path or media_folder_path):
    with st.spinner("Analyzing reference and generating auto edit..."):
        with tempfi
