import os
import subprocess
import tempfile
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import clip
import cv2
import whisper
import streamlit as st
import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import requests
from bs4 import BeautifulSoup
import re
import zipfile

# =========================================
# 1. Upload Section with Google Drive Support
# =========================================

st.set_page_config(page_title="Smart Auto Edit", layout="wide")
st.title("🎬 Smart Auto Edit – Reference-Based Editor")

# 1.1 Reference Video Input
st.subheader("1. Reference Video Input")
ref_method = st.radio(
    "Choose input for reference video:",
    ("Upload from Device", "Google Drive Link")
)
reference_path = None

if ref_method == "Upload from Device":
    reference_file = st.file_uploader("Upload Reference Video", type=["mp4", "mov"])
    if reference_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(reference_file.name)[1]) as tmp_ref:
            tmp_ref.write(reference_file.read())
            reference_path = tmp_ref.name
else:
    reference_drive_link = st.text_input("Paste Google Drive share link for reference video")

# 1.2 Raw Media Input
st.subheader("2. Raw Media Input")
media_method = st.radio(
    "Choose input for raw media clips:",
    ("Upload ZIP from Device", "Google Drive Folder Link")
)
media_folder_path = None

if media_method == "Upload ZIP from Device":
    media_zip = st.file_uploader("Upload Raw Media ZIP (.zip)", type=["zip"])
    if media_zip:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, media_zip.name)
            with open(zip_path, "wb") as f:
                f.write(media_zip.read())
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)
            media_folder_path = tmp_dir
else:
    media_drive_link = st.text_input("Paste Google Drive folder share link for raw media")

def download_from_drive(drive_url, output_path):
    # Extract file ID
    match = re.search(r"(?:file/d/|id=)([\w-]{10,})", drive_url)
    if not match:
        st.error("Invalid Google Drive link.")
        return None
    file_id = match.group(1)

    # Initial download URL
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(download_url, stream=True)

    # Bypass large-file warning if needed
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            download_url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(download_url, stream=True)
            break

    # Determine correct extension from Content-Type
    content_type = response.headers.get("Content-Type", "")
    if "video/mp4" in content_type:
        ext = ".mp4"
    elif "video/quicktime" in content_type:
        ext = ".mov"
    else:
        # Fallback to output_path's extension or default to .mp4
        ext = os.path.splitext(output_path)[1] if os.path.splitext(output_path)[1] else ".mp4"

    # Build actual output path with correct extension
    base, _ = os.path.splitext(output_path)
    actual_path = base + ext

    # Write to file
    try:
        with open(actual_path, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        st.error(f"Failed to write file: {e}")
        return None

    return actual_path


def scrape_drive_folder(folder_url, dest_folder):
    response = requests.get(folder_url)
    soup = BeautifulSoup(response.text, "html.parser")
    for a_tag in soup.find_all("a"):
        href = a_tag.get("href", "")
        if "uc?id=" in href or "file/d/" in href:
            fid = None
            if "uc?id=" in href:
                fid = href.split("uc?id=")[-1].split("&")[0]
            elif "file/d/" in href:
                fid = href.split("file/d/")[-1].split("/")[0]
            if fid:
                dl_url = f"https://drive.google.com/uc?export=download&id={fid}"
                resp = requests.get(dl_url, stream=True)
                ctype = resp.headers.get("Content-Type", "")
                ext = None
                if "video/mp4" in ctype:
                    ext = "mp4"
                elif "video/quicktime" in ctype:
                    ext = "mov"
                if ext:
                    out_path = os.path.join(dest_folder, f"file_{fid}.{ext}")
                    with open(out_path, "wb") as out_f:
                        for chunk in resp.iter_content(32768):
                            if chunk:
                                out_f.write(chunk)
    return dest_folder

# After user inputs, fetch files from Drive
if ref_method == "Google Drive Link" and 'reference_drive_link' in locals() and reference_drive_link:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        reference_path = download_from_drive(reference_drive_link, tmp.name)

if media_method == "Google Drive Folder Link" and 'media_drive_link' in locals() and media_drive_link:
    media_folder_path = tempfile.mkdtemp()
    scrape_drive_folder(media_drive_link, media_folder_path)

# Check if both inputs are ready
if not reference_path or not media_folder_path:
    st.warning("Please provide both a reference video and raw media clips before proceeding.")
    st.stop()

# ------------------------------------
# Section 2: Analysis Utilities
# ------------------------------------

st.subheader("3. Analysis: Audio Transcription, Scene Detection, Visual Embeddings")

# 2.1 Whisper Audio Transcription
def transcribe_audio(video_path):
    # Transcribe and classify segments as speech or music
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    segments = result.get("segments", [])
    transcription = []
    for seg in segments:
        text = seg.get("text", "")
        if any(keyword in text.lower() for keyword in ["uh", "um", "like", "you know"]):
            seg_type = "speech"
        elif any(char.isalpha() for char in text) and text.strip().endswith('.'):
            seg_type = "speech"
        else:
            seg_type = "music"
        transcription.append({
            "start": seg["start"],
            "end": seg["end"],
            "type": seg_type,
            "text": text
        })
    return transcription

# 2.2 CLIP Embedding Extraction (middle frame)
def extract_middle_frame_embedding(video_path):
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_input = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(img_input)
    return emb[0].numpy()

# 2.3 Scene Detection Using PySceneDetect
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=35.0))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

# 2.4 GPT-4 Edit Style Guidance
def get_edit_style(shots_info):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    messages = [
        {"role": "system", "content": "You are a professional video editor assistant. Analyze the following shot timing information and recommend pacing and transitions."},
        {"role": "user", "content": f"Here are the shots with start/end times: {json.dumps(shots_info)}. Provide a JSON response with keys 'order' (list of shot indices), 'durations' (list of durations in seconds), and 'transitions' (list of transition types)."}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.5
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"GPT-4 request failed: {e}")
        # Fallback: use original order and durations
        return {"order": list(range(len(shots_info))), "durations": [end - start for start, end in shots_info], "transitions": ["cut"] * len(shots_info)}

# ------------------------------------
# Section 3: Auto-Edit Pipeline
# ------------------------------------

st.subheader("4. Auto-Edit Generation and Export")

# 3.1 Auto-Edit Function
@st.cache_resource
def auto_edit(reference_path, media_folder_path):
    try:
        scenes = detect_scenes(reference_path)
    except Exception as e:
        st.error(f"Scene detection failed: {e}")
        return []
    try:
        transcription = transcribe_audio(reference_path)
    except Exception as e:
        st.error(f"Audio transcription failed: {e}")
        transcription = []
    try:
        ref_emb = extract_middle_frame_embedding(reference_path)
    except Exception as e:
        st.error(f"CLIP embedding extraction failed: {e}")
        ref_emb = None

    st.info("Starting auto-edit process...")
    progress = st.progress(0)
    total_shots = len(scenes)
    output_clips = []

    ref_cap = cv2.VideoCapture(reference_path)
    ref_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_cap.release()
    ref_ratio = ref_width / ref_height

    for idx, (start_time, end_time) in enumerate(scenes):
        try:
            best_score = float("-inf")
            best_path = None
            for root, _, files in os.walk(media_folder_path):
                for file in files:
                    if file.lower().endswith((".mp4", ".mov")):
                        path = os.path.join(root, file)
                        emb = None
                        try:
                            emb = extract_middle_frame_embedding(path)
                        except Exception:
                            continue
                        if emb is None or ref_emb is None:
                            continue
                        score = np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb) + 1e-6)
                        if score > best_score:
                            best_score = score
                            best_path = path
            if not best_path:
                st.warning(f"No matching clip found for shot {idx+1}")
                continue
            frame = None
            try:
                frame = cv2.imread(best_path)
            except Exception as e:
                st.warning(f"Failed to read frame for {best_path}: {e}")
                continue
            crop_box = None
            try:
                crop_box = detect_subject_crop(frame, ref_ratio)
            except RuntimeError as e:
                st.error(f"Smart crop model missing: {e}")
                return []
            if crop_box:
                x1, y1, x2, y2 = crop_box
                vf = f"crop={x2-x1}:{y2-y1}:{x1}:{y1},scale={ref_width}:{ref_height}"
            else:
                vf = f"scale={ref_width}:{ref_height}"

                        # Determine audio overlay type using raw clip transcription
            audio_flag = "music"
            try:
                raw_transcription = transcribe_audio(best_path)
                # If the first segment in the raw clip is speech, classify as speech
                if raw_transcription and raw_transcription[0]["type"] == "speech":
                    audio_flag = "speech"
            except Exception as e:
                st.warning(f"Raw clip transcription failed: {e}")

            temp_vid = f"temp_{idx+1:03d}.mp4"
            cmd_video = [
                "ffmpeg", "-y", "-i", best_path,
                "-vf", vf,
                "-ss", str(start_time), "-t", str(end_time - start_time),
                "-an", temp_vid
            ]

                "ffmpeg", "-y", "-i", best_path,
                "-vf", vf,
                "-ss", str(start_time), "-t", str(end_time - start_time),
                "-an", temp_vid
            ]
            subprocess.run(cmd_video)

            out_name = f"clip_{idx+1:03d}.mp4"
            if audio_flag == "speech":
                temp_audio = f"audio_{idx+1:03d}.mp3"
                cmd_audio = [
                    "ffmpeg", "-y", "-i", reference_path,
                    "-ss", str(start_time), "-t", str(end_time - start_time),
                    "-q:a", "0", "-map", "a", temp_audio
                ]
                subprocess.run(cmd_audio)
                cmd_merge = [
                    "ffmpeg", "-y", "-i", temp_vid, "-i", temp_audio,
                    "-c:v", "copy", "-c:a", "aac", "-shortest", out_name
                ]
                subprocess.run(cmd_merge)
                os.remove(temp_vid)
                os.remove(temp_audio)
            else:
                cmd_passthrough = [
                    "ffmpeg", "-y", "-i", best_path,
                    "-vf", vf,
                    "-ss", str(start_time), "-t", str(end_time - start_time),
                    out_name
                ]
                subprocess.run(cmd_passthrough)
            output_clips.append(out_name)
        except Exception as shot_e:
            st.warning(f"Error processing shot {idx+1}: {shot_e}")
            continue
        progress.progress((idx + 1) / total_shots)

    try:
        with open("inputs.txt", "w") as f:
            for clip in output_clips:
                f.write(f"file '{clip}'
")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "inputs.txt", "-c", "copy", "final_output.mp4"])
    except Exception as e:
        st.error(f"Concatenation failed: {e}")

    try:
        root = ET.Element("project")
        timeline = ET.SubElement(root, "timeline")
        for idx, (start_time, end_time) in enumerate(scenes):
            duration_ms = int((end_time - start_time) * 1000)
            clip_name = f"clip_{idx+1:03d}.mp4"
            ET.SubElement(timeline, "clip", {"src": clip_name, "start": "0", "duration": str(duration_ms)})
        tree = ET.ElementTree(root)
        tree.write("capcut_project.xml")
    except Exception as e:
        st.error(f"CapCut XML export failed: {e}")

    
    # 3.2 Cleanup temporary files
    temp_files = ["inputs.txt"] + [f"clip_{i+1:03d}.mp4" for i in range(len(scenes))] + [f"temp_{i+1:03d}.mp4" for i in range(len(scenes))] + [f"audio_{i+1:03d}.mp3" for i in range(len(scenes))]
    for fpath in temp_files:
        try:
            if os.path.exists(fpath): os.remove(fpath)
        except:
            pass
    return output_clips

# 4.1 Streamlit UI Actions: Start and Download
if st.button("Start Auto Edit"):
    with st.spinner("Generating auto edit, please wait..."):
        clips = auto_edit(reference_path, media_folder_path)
        st.success("Auto edit complete.")
        st.video("final_output.mp4")
        st.download_button("Download Final Video", data=open("final_output.mp4","rb"), file_name="final_output.mp4")
        st.download_button("Download CapCut XML", data=open("capcut_project.xml","rb"), file_name="capcut_project.xml")

# ------------------------------------
# Section 4: Template/Batch Processing
# ------------------------------------

TEMPLATES_FILE = "templates.json"

# 4.1 Save Current Template
def save_template(name, ref_path, media_path):
    templates = {}
    if os.path.exists(TEMPLATES_FILE):
        with open(TEMPLATES_FILE, "r") as tf:
            templates = json.load(tf)
    templates[name] = {"reference": ref_path, "media": media_path}
    with open(TEMPLATES_FILE, "w") as tf:
        json.dump(templates, tf)
    st.success(f"Template '{name}' saved.")

# 4.2 Load Template
def load_templates():
    if not os.path.exists(TEMPLATES_FILE):
        return {}
    with open(TEMPLATES_FILE, "r") as tf:
        return json.load(tf)

# Template UI
st.subheader("5. Template / Batch Processing")
template_name = st.text_input("Template Name (to save current setup)")
if st.button("Save Template") and template_name:
    save_template(template_name, reference_path, media_folder_path)

existing = load_templates()
if existing:
    chosen = st.selectbox("Load existing template", list(existing.keys()))
    if st.button("Apply Template"):
        tpl = existing[chosen]
        reference_path = tpl["reference"]
        media_folder_path = tpl["media"]
        st.success(f"Template '{chosen}' applied.")

# End of App
