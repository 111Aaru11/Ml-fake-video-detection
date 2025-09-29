import os
import argparse
import json
from utils.video_utils import extract_frames, extract_audio_as_wav, load_audio_features, save_plot
from utils.analysis_utils import analyze_video, plot_timeseries

def aggregate_scores(metrics):
    """
    Combine metrics into a single heuristic 'fake_score' in [0,1].
    Higher -> more likely fake.
    Components:
      - very low blink rate -> suspicious
      - low lip-sync corr -> suspicious
      - very low head jitter -> suspicious (too still)
      - unusually high freq_mean -> suspicious
    """
    import numpy as np
    # blink rate: expected for real people ~ 0.2 - 0.4 blinks/sec (rough). If extremely low, suspicious.
    blink_rate = metrics.get("blink_rate", 0.0)
    # map blink_rate: if blink_rate < 0.05 -> suspicious
    blink_score = max(0.0, min(1.0, (0.05 - blink_rate) / 0.05)) if blink_rate < 0.05 else 0.0

    # lip-sync correlation: expected moderate->high for real; if negative or small -> suspicious
    corr = metrics.get("lip_sync_corr")
    if corr is None or (isinstance(corr, float) and (abs(corr) < 0.2 or (corr != corr))):
        lip_score = 1.0  # strongly suspicious if no correlation / low
    else:
        lip_score = max(0.0, 1.0 - min(1.0, corr))  # lower corr -> higher score

    # head jitter: very low jitter suspicious
    jitter = metrics.get("head_jitter", 0.0)
    jitter_score = 0.0
    if jitter < 1.0:
        jitter_score = max(0.0, min(1.0, (1.0 - jitter) / 1.0))

    # freq fingerprint: larger values suspicious (heuristic)
    freq_mean = metrics.get("freq_mean", 0.0)
    freq_score = max(0.0, min(1.0, (freq_mean - 0.07) / 0.2))  # tuneable

    # weighted sum
    fake_score = (0.35 * blink_score) + (0.35 * lip_score) + (0.15 * jitter_score) + (0.15 * freq_score)
    fake_score = float(max(0.0, min(1.0, fake_score)))
    verdict = "FAKE (likely AI-generated)" if fake_score > 0.5 else "REAL (likely real capture)"
    return {
        "fake_score": fake_score,
        "verdict": verdict,
        
        "components": {
            "blink_score": blink_score,
            "lip_score": lip_score,
            "jitter_score": jitter_score,
            "freq_score": freq_score
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Fake video detection (heuristic)")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--out", default="outputs", help="Output folder")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio-based lip-sync check")
    args = parser.parse_args()

    video_path = args.video
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    frames, fps = extract_frames(video_path)
    audio_rms = None
    audio_times = None
    wav_path = os.path.join(out_dir, "extracted_audio.wav")
    if not args.no_audio:
        try:
            extract_audio_as_wav(video_path, wav_path)
            _, _, audio_rms, audio_times = load_audio_features(wav_path, sr=16000)
            import numpy as np
            audio_times = np.array(audio_times)
            audio_rms = np.array(audio_rms)
        except Exception as e:
            print("Audio extraction failed:", e)
            audio_rms = None
            audio_times = None

    metrics = analyze_video(frames, fps, audio_rms=audio_rms, audio_times=audio_times, verbose=True)

    # aggregate to final score
    agg = aggregate_scores(metrics)

    # produce plots and JSON report
    report = {**metrics, **agg}
    # JSON serialization: convert numpy arrays to lists, floats
    def convert(obj):
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (float, int, str, type(None))):
            return obj
        try:
            return float(obj)
        except:
            return str(obj)

    out_json_path = os.path.join(out_dir, "report.json")
    with open(out_json_path, "w") as f:
        json.dump({k: convert(v) for k, v in report.items()}, f, indent=2)

    # make plot
    plot_path = os.path.join(out_dir, "plots", "timeseries.png")
    plot_timeseries(plot_path, metrics["frame_times"], metrics["ear"], metrics["mar"],
                    audio_energy=audio_rms, audio_times=audio_times, lip_sync_corr=metrics.get("lip_sync_corr"))

    print("\n--- Detection Report ---")
    print("Video:", video_path)
    print("Fake score: {:.3f}".format(agg["fake_score"]))
    print("Verdict:", agg["verdict"])
    print("Component scores:", agg["components"])
    print("Report saved to:", out_json_path)
    print("Plot saved to:", plot_path)

if __name__ == "__main__":
    main()
