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

# Upload Section
st.subheader("1. Upload Reference Video and Raw Media")
reference_file = st.file_uploader("Upload Edited Reference Video", type=["mp4", "mov"])
media_zip = st.file_uploader("Upload Raw Media Clips (.zip)", type=["zip"])

reference_path = None
media_folder_path = None

if reference_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(reference_file.name)[1]) as tmp_ref:
        tmp_ref.write(reference_file.read())
        reference_path = tmp_ref.name

if media_zip:
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, media_zip.name)
        with open(zip_path, "wb") as f:
            f.write(media_zip.read())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        media_folder_path = tmp_dir

# Smart Crop

def detect_subject_crop(frame, target_ratio):
    model_file = "MobileNetSSD_deploy.caffemodel"
    config_file = "MobileNetSSD_deploy.prototxt"
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        raise RuntimeError("DNN model files are missing.")
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    (h, w) = frame.shape[:2]
    best_box = None
    max_confidence = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if confidence > max_confidence:
                max_confidence = confidence
                best_box = (startX, startY, endX, endY)

    if best_box is not None:
        startX, startY, endX, endY = best_box
        subject_center_x = (startX + endX) // 2
        subject_center_y = (startY + endY) // 2

        if target_ratio > 1:
            crop_w = min(w, int(h * target_ratio))
            crop_h = h
        else:
            crop_w = w
            crop_h = min(h, int(w / target_ratio))

        x1 = max(0, subject_center_x - crop_w // 2)
        y1 = max(0, subject_center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        return x1, y1, x2, y2
    else:
        return None

# Whisper Transcription

def transcribe_audio(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result

# Extract Middle Frame

def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    success, frame = cap.read()
    cap.release()
    if success:
        return frame
    return None

# Auto Edit Generation

def auto_edit(reference_path, media_folder_path):
    video_manager = VideoManager([reference_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    video_manager.release()

    model, preprocess = clip.load("ViT-B/32", device="cpu")

    cap = cv2.VideoCapture(reference_path)
    ref_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    ref_ratio = ref_width / ref_height

    shots = []
    for start_time, end_time in scenes:
        shots.append({"start": start_time.get_seconds(), "end": end_time.get_seconds()})

    output_clips = []
    for i, shot in enumerate(shots):
        for root, _, files in os.walk(media_folder_path):
            for file in files:
                if file.lower().endswith((".mp4", ".mov")):
                    path = os.path.join(root, file)
                    frame = extract_middle_frame(path)
                    if frame is None:
                        continue

                    crop_box = detect_subject_crop(frame, ref_ratio)
                    if crop_box is None:
                        crop_filter = f"scale=1080:1920"
                    else:
                        x1, y1, x2, y2 = crop_box
                        crop_filter = f"crop={x2-x1}:{y2-y1}:{x1}:{y1},scale=1080:1920"

                    output_name = f"output_clip_{i+1:03d}.mp4"
                    cmd = [
                        "ffmpeg", "-y", "-i", path,
                        "-vf", crop_filter,
                        "-ss", str(shot["start"]), "-t", str(shot["end"] - shot["start"]),
                        output_name
                    ]
                    subprocess.run(cmd)
                    output_clips.append(output_name)

    generate_capcut_xml(output_clips)

# CapCut XML Export

def generate_capcut_xml(output_clips):
    root = ET.Element("project")
    timeline = ET.SubElement(root, "timeline")
    for clip in output_clips:
        ET.SubElement(timeline, "clip", {
            "src": clip,
            "start": "0",
            "duration": "5000"
        })
    tree = ET.ElementTree(root)
    tree.write("capcut_project.xml")

if st.button("Start Auto Edit"):
    if reference_path and media_folder_path:
        auto_edit(reference_path, media_folder_path)
        st.success("Auto edit complete. Output clips and CapCut XML generated.")

        with open("capcut_project.xml", "rb") as f:
            st.download_button("Download CapCut XML", f, file_name="capcut_project.xml")
