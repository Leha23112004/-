import os
import json
import subprocess
from typing import List

import requests
from googletrans import Translator
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
from yt_dlp import YoutubeDL
import whisper


def fetch_popular_shorts(api_key: str, max_results: int = 5) -> List[str]:
    """Fetch popular YouTube shorts using YouTube Data API."""
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "maxResults": max_results,
        "order": "viewCount",
        "type": "video",
        "videoDuration": "short",
        "fields": "items(id/videoId)"
    }
    headers = {"Accept": "application/json"}
    params["key"] = api_key
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return [item["id"]["videoId"] for item in data.get("items", [])]


def download_video(video_id: str, output_dir: str = "downloads") -> str:
    """Download a YouTube video using yt_dlp."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}.mp4")
    ydl_opts = {
        "format": "best[height<=720]",
        "outtmpl": output_path,
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    return output_path


def transcribe_audio(video_path: str) -> str:
    """Transcribe audio using OpenAI Whisper."""
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result["text"]


def translate_text(text: str, dest_lang: str = "ru") -> str:
    """Translate text using googletrans."""
    translator = Translator()
    return translator.translate(text, dest=dest_lang).text


def synthesize_speech(text: str, lang: str = "ru", output="tts.mp3") -> str:
    """Generate speech audio from text."""
    tts = gTTS(text=text, lang=lang)
    tts.save(output)
    return output


def add_subtitles(video_path: str, transcription: str, tts_path: str, output: str) -> str:
    """Overlay subtitles and TTS audio onto a video."""
    video = VideoFileClip(video_path)
    audio = AudioFileClip(tts_path)
    subtitle = TextClip(transcription, fontsize=24, color='white', method='caption', size=(video.w, None))
    subtitle = subtitle.set_position(('center', 'bottom')).set_duration(video.duration)
    final = CompositeVideoClip([video, subtitle])
    final = final.set_audio(audio.set_duration(video.duration))
    final.write_videofile(output, codec="libx264", audio_codec="aac")
    return output


def process_video(video_id: str, dest_lang: str = "ru"):
    video_path = download_video(video_id)
    text = transcribe_audio(video_path)
    translated_text = translate_text(text, dest_lang)
    tts_path = synthesize_speech(translated_text, lang=dest_lang)
    output_video = f"processed_{video_id}.mp4"
    add_subtitles(video_path, translated_text, tts_path, output_video)
    print(f"Processed video saved to {output_video}")


def main():
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY environment variable not set")
    video_ids = fetch_popular_shorts(api_key)
    for vid in video_ids:
        process_video(vid)


if __name__ == "__main__":
    main()
