import yt_dlp
from moviepy.video.io.VideoFileClip import VideoFileClip
from faster_whisper import WhisperModel
import requests
import os
from django.conf import settings
import subprocess

import subprocess

def reencode_video(input_path):
    print(f"[DEBUG] Starting re-encoding for video: {input_path}")

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file does not exist: {input_path}")
        return None
    else:
        print(f"[DEBUG] Input file found: {input_path}")

    # Generate output path in the same folder
    output_path = input_path.replace(".mp4", "_fixed.mp4")
    print(f"[DEBUG] Output path set to: {output_path}")

    command = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libx264", "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ]
    print(f"[DEBUG] Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print(f"[DEBUG] Re-encoding successful. Output file: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Re-encoding failed with error: {e}")
        return None

    # Confirm that the output file now exists
    if os.path.exists(output_path):
        print(f"[DEBUG] Verified output file exists: {output_path}")
    else:
        print(f"[ERROR] Output file still missing: {output_path}")
    
    return output_path




def download_google_drive_file(url, title):
    output_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    sanitized_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)  # Sanitize title
    output_file_path = os.path.join(output_dir, f"{sanitized_title}.mp4")

    if "id=" in url:
        file_id = url.split("id=")[1]
    elif "file/d/" in url:
        file_id = url.split("file/d/")[1].split("/")[0]
    else:
        raise ValueError("Invalid Google Drive URL")

    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"

    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        with open(output_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return output_file_path
    else:
        raise Exception(f"Failed to download file from Google Drive. Status code: {response.status_code}")





def download_dropbox_file(url, title):
    output_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    sanitized_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)  # Sanitize title
    output_file_path = os.path.join(output_dir, f"{sanitized_title}.mp4")

    if "?dl=0" in url:
        direct_url = url.replace("?dl=0", "?dl=1")
    elif "?dl=1" not in url:
        direct_url = f"{url}?dl=1"
    else:
        direct_url = url

    response = requests.get(direct_url, stream=True)
    if response.status_code == 200:
        with open(output_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return output_file_path
    else:
        raise Exception(f"Failed to download file from Dropbox. Status code: {response.status_code}")




def transcribe_audio_with_timestamps(audio_path):
    """
    Transcribes an audio file and returns both full text and segments with timestamps.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        tuple: Full transcription text and a list of segments with timestamps.
               Each segment is a dictionary with 'start', 'end', and 'text'.
    """
    model = WhisperModel("base", device="cpu")  # Use "cuda" if you have a GPU
    segments, _ = model.transcribe(audio_path)

    full_transcript = []
    timestamped_segments = []

    for segment in segments:
        start_time = segment.start  # Start time in seconds
        end_time = segment.end      # End time in seconds
        text = segment.text.strip()  # Transcribed text

        full_transcript.append(text)
        timestamped_segments.append({
            "start": start_time,
            "end": end_time,
            "text": text,
        })

    return " ".join(full_transcript), timestamped_segments
def extract_audio(video_path):
    """
    Extracts the audio from a video file and saves it as a .wav file.
    Args:
        video_path (str): The path to the video file.
    Returns:
        str: The path to the extracted audio file.
    """
    print(f"[DEBUG] Starting audio extraction for: {video_path}")

    # Verify that the video file exists
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return None
    else:
        print(f"[DEBUG] Video file found: {video_path}")

    # Define the audio output path
    audio_path = video_path.replace(".mp4", ".wav")
    print(f"[DEBUG] Audio output will be: {audio_path}")

    try:
        clip = VideoFileClip(video_path)
        # Extract and write the audio file
        clip.audio.write_audiofile(audio_path)
        print(f"[DEBUG] Audio extraction successful: {audio_path}")
    except Exception as e:
        print(f"[ERROR] Audio extraction failed: {e}")
        return None

    # Confirm that the audio file now exists
    if os.path.exists(audio_path):
        print(f"[DEBUG] Verified audio file exists: {audio_path}")
    else:
        print(f"[ERROR] Audio file does not exist after extraction: {audio_path}")

    return audio_path

def download_youtube_video(url, title):
    """
    Downloads a YouTube video in MP4 format.
    Args:
        url (str): The URL of the YouTube video.
        title (str): The title to save the video as.
    Returns:
        str: The path to the downloaded video file.
    """
    output_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # Use title as the output file name
    sanitized_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)  # Sanitize title
    output_file = f"{output_dir}/{sanitized_title}.mp4"

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',  # Force MP4 format
        'outtmpl': output_file,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4'  # Convert to MP4 if needed
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
    return output_file
