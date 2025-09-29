import cv2
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
from pydub import AudioSegment

def extract_frames(video_path):
    """
    Returns list of frames (BGR) and fps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for _ in tqdm(range(total), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())
    cap.release()
    return frames, fps

def extract_audio_as_wav(video_path, out_wav_path):
    """
    Use pydub to export audio from video to wav. Requires ffmpeg installed.
    """
    audio = AudioSegment.from_file(video_path)
    audio.export(out_wav_path, format="wav")
    return out_wav_path

def load_audio_features(wav_path, sr=16000):
    """
    Loads audio with librosa and returns RMS energy per sample and sr.
    """
    y, sr = librosa.load(wav_path, sr=sr)
    # RMS energy
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop_length).flatten()
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    return y, sr, rms, times

def save_plot(fig, path):
    import os
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(path, bbox_inches='tight', dpi=150)
