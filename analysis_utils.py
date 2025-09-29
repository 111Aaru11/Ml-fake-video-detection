import mediapipe as mp
import cv2
import numpy as np
from .landmark_utils import (landmark_list_to_np, LEFT_EYE, RIGHT_EYE, eye_aspect_ratio,
                             UPPER_LIP, LOWER_LIP, normalized_face_width, mouth_opening_ratio)
from scipy.fftpack import fft2, fftshift
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                          max_num_faces=1,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)

def analyze_video(frames, fps, audio_rms=None, audio_times=None, verbose=False):
    """
    frames: list of BGR frames
    fps: frames per second
    audio_rms: numpy array of audio RMS energy per audio frame (librosa)
    audio_times: times for each rms value (seconds)
    Returns: dictionary with metrics and time-series.
    """
    ear_series = []
    mar_series = []
    face_centers = []
    freq_scores = []
    frame_times = []
    img_shape = frames[0].shape

    for i, frame in enumerate(frames):
        time_sec = i / fps
        frame_times.append(time_sec)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)
        if not results.multi_face_landmarks:
            ear_series.append(None)
            mar_series.append(None)
            face_centers.append(None)
            freq_scores.append(None)
            continue
        lm = results.multi_face_landmarks[0].landmark
        pts = landmark_list_to_np(lm, img_shape)

        # EAR left + right
        left_eye_pts = pts[LEFT_EYE]
        right_eye_pts = pts[RIGHT_EYE]
        ear_l = eye_aspect_ratio(left_eye_pts)
        ear_r = eye_aspect_ratio(right_eye_pts)
        ear = (ear_l + ear_r) / 2.0
        ear_series.append(ear)

        # mouth opening
        face_w = normalized_face_width(pts)
        upper = pts[UPPER_LIP]
        lower = pts[LOWER_LIP]
        mar = mouth_opening_ratio(upper, lower, face_w)
        mar_series.append(mar)

        # face center for head jitter
        center = np.mean(pts, axis=0)
        face_centers.append(center)

        # frequency fingerprint: crop face, convert to grayscale, fft
        # take center square around face center
        h, w = frame.shape[:2]
        cx, cy = int(center[0]), int(center[1])
        half = int(min(h, w) * 0.2)
        x1, y1 = max(0, cx-half), max(0, cy-half)
        x2, y2 = min(w, cx+half), min(h, cy+half)
        face_patch = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if face_patch.size == 0:
            freq_scores.append(None)
        else:
            f = fft2(face_patch)
            fshift = fftshift(f)
            magnitude = np.abs(fshift)
            # compute ratio of high frequency energy to total energy
            center_x, center_y = magnitude.shape[1]//2, magnitude.shape[0]//2
            cy, cx = magnitude.shape
            mask_radius = int(min(cx, cy) * 0.15)
            Y, X = np.ogrid[:cy, :cx]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            high_freq_mask = dist_from_center > mask_radius
            total = magnitude.sum()
            if total == 0:
                freq_scores.append(0.0)
            else:
                high_ratio = magnitude[high_freq_mask].sum() / total
                freq_scores.append(high_ratio)

    # postprocess time-series: convert None to np.nan
    import numpy as _np
    ear_array = _np.array([v if v is not None else _np.nan for v in ear_series], dtype=float)
    mar_array = _np.array([v if v is not None else _np.nan for v in mar_series], dtype=float)
    freq_array = _np.array([v if v is not None else _np.nan for v in freq_scores], dtype=float)
    centers = _np.array([c if c is not None else [np.nan, np.nan] for c in face_centers], dtype=float)

    # blinking heuristic: compute average EAR and count low-ear frames as closed eyes
    ear_mean = float(_np.nanmean(ear_array)) if not _np.all(_np.isnan(ear_array)) else 0.0
    blink_count = int(_np.nansum((ear_array < 0.22).astype(float))) if ear_array.size else 0
    blink_rate = blink_count / (len(frames) / fps)  # blinks per second (very rough)

    # head jitter: compute frame-to-frame movement magnitude
    diffs = _np.linalg.norm(_np.diff(centers, axis=0), axis=1)
    head_jitter = float(_np.nanmean(diffs)) if diffs.size else 0.0

    # lip-sync correlation: align audio_rms (times) with frame_times -> sample audio energy per video frame
    lip_sync_corr = None
    if audio_rms is not None and audio_times is not None:
        # compute average audio energy per video frame
        audio_energy_per_frame = []
        for t in frame_times:
            # find nearest audio_time index
            idx = (abs(audio_times - t)).argmin()
            audio_energy_per_frame.append(audio_rms[idx])
        audio_energy_per_frame = _np.array(audio_energy_per_frame)
        # compute correlation between mouth opening and audio energy
        # only for indices where both exist
        valid = ~_np.isnan(mar_array) & ~_np.isnan(audio_energy_per_frame)
        try:
            if valid.sum() > 10:
                corr, _ = pearsonr(mar_array[valid], audio_energy_per_frame[valid])
                lip_sync_corr = float(corr)
            else:
                lip_sync_corr = float(_np.nan)
        except Exception:
            lip_sync_corr = float(_np.nan)

    # freq fingerprint score (mean)
    freq_mean = float(_np.nanmean(freq_array)) if not _np.all(_np.isnan(freq_array)) else 0.0

    results = {
        "frame_times": _np.array(frame_times),
        "ear": ear_array,
        "mar": mar_array,
        "freq": freq_array,
        "face_centers": centers,
        "ear_mean": ear_mean,
        "blink_count": blink_count,
        "blink_rate": blink_rate,
        "head_jitter": head_jitter,
        "lip_sync_corr": lip_sync_corr,
        "freq_mean": freq_mean
    }

    if verbose:
        print("EAR mean:", results["ear_mean"])
        print("Blink count (low-ear frames):", results["blink_count"])
        print("Head jitter:", results["head_jitter"])
        print("Lip-sync corr:", results["lip_sync_corr"])
        print("Freq mean:", results["freq_mean"])
    return results

def plot_timeseries(out_path, frame_times, ear, mar, audio_energy=None, audio_times=None, lip_sync_corr=None):
    """
    Creates and saves a figure showing ear, mar and optionally audio energy.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(frame_times, ear, label="EAR (eye aspect ratio)")
    ax[0].legend(loc='upper right')
    ax[1].plot(frame_times, mar, label="MAR (mouth opening)",)
    ax[1].legend(loc='upper right')
    if audio_energy is not None and audio_times is not None:
        # resample audio energy to frame times (nearest)
        audio_vals = [audio_energy[(abs(audio_times - t)).argmin()] for t in frame_times]
        ax[2].plot(frame_times, audio_vals, label="Audio RMS energy")
    else:
        ax[2].plot(frame_times, np.zeros_like(frame_times), label="No audio")
    ax[2].legend(loc='upper right')
    plt.xlabel("Time (s)")
    plt.suptitle(f"Lip-sync corr: {lip_sync_corr:.3f}" if lip_sync_corr is not None else "Lip-sync: N/A")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
