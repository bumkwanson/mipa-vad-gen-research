"""
오디오 분석 모듈 (vad-gen)
- faster-whisper: 대사 STT + 타임스탬프 (영어)
- silero-VAD: 발화/침묵 구간 분리

영상 파일에서 오디오를 추출해 두 가지를 반환:
  1) 발화 세그먼트 (start, end, text)
  2) 침묵 구간 (start, end)
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad


def extract_audio(video_path, sr=16000):
    """영상에서 16kHz 모노 wav 추출 (임시 파일 경로 반환)"""
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg가 설치되어 있지 않습니다. README의 설치 절차를 확인하세요."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        Path(tmp_path).unlink(missing_ok=True)
        detail = (
            result.stderr.strip().splitlines()[-1]
            if result.stderr.strip()
            else "알 수 없는 오류"
        )
        raise RuntimeError(f"영상에서 오디오를 추출하지 못했습니다: {detail}")
    return tmp_path


class AudioAnalyzer:
    def __init__(
        self,
        whisper_model="large-v3-turbo",
        device="auto",
        language="en",
        compute_type=None,
    ):
        from faster_whisper import WhisperModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 장치를 요청했지만 PyTorch가 GPU를 인식하지 못합니다."
            )
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"지원하지 않는 장치입니다: {device}")

        self.language = language
        self.device = device
        compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.whisper = WhisperModel(
            whisper_model, device=device, compute_type=compute_type
        )

        # 설치한 silero-vad 패키지의 모델을 사용해 실행마다 코드 버전이 바뀌지 않게 한다.
        torch.set_num_threads(1)
        self.vad_model = load_silero_vad(onnx=False)

    def transcribe(self, audio_path):
        """대사 STT: [{'start','end','text'}, ...]"""
        segments, info = self.whisper.transcribe(
            audio_path,
            language=self.language,
            beam_size=5,
            word_timestamps=True,  # 단어별 타임스탬프
            vad_filter=True,  # 무음 기준으로 세그먼트 분할
        )
        out = []
        for seg in segments:
            words = []
            if seg.words:
                words = [
                    {"start": w.start, "end": w.end, "word": w.word} for w in seg.words
                ]
            out.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "words": words,
                }
            )
        return out, info

    def detect_speech_silence(
        self, audio_path, threshold=0.7, sr=16000, min_silence=1.5
    ):
        """
        발화/침묵 구간 검출.
        반환: (speech_segments, silence_segments)
              각각 [{'start','end'}, ...]
        """
        wav_np, file_sr = sf.read(audio_path, dtype="float32")
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)
        if file_sr != sr:
            raise ValueError(f"샘플레이트 불일치: {file_sr} != {sr}")
        wav = torch.from_numpy(wav_np)

        ts = get_speech_timestamps(
            wav, self.vad_model, sampling_rate=sr, threshold=threshold
        )

        speech = [{"start": t["start"] / sr, "end": t["end"] / sr} for t in ts]

        # 침묵 = 발화 구간 사이 + 앞뒤 여백
        total_dur = len(wav_np) / sr
        silence = []
        prev_end = 0.0
        for s in speech:
            gap = s["start"] - prev_end
            if gap > 0.3 and gap >= min_silence:  # 0.3초 이상만 의미있는 침묵으로
                silence.append({"start": prev_end, "end": s["start"]})
            prev_end = s["end"]
        trailing_gap = total_dur - prev_end
        if trailing_gap > 0.3 and trailing_gap >= min_silence:
            silence.append({"start": prev_end, "end": total_dur})

        return speech, silence

    def analyze(self, video_path):
        """영상 -> 오디오 추출 -> STT + VAD 한번에"""
        audio_path = extract_audio(video_path)
        try:
            transcript, info = self.transcribe(audio_path)
            speech, silence = self.detect_speech_silence(audio_path)
            return {
                "transcript": transcript,
                "speech": speech,
                "silence": silence,
                "duration": info.duration,
                "language": info.language,
            }
        finally:
            Path(audio_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python audio_analysis.py <영상경로>")
        sys.exit(1)

    analyzer = AudioAnalyzer()
    res = analyzer.analyze(sys.argv[1])

    print(f"언어: {res['language']}, 길이: {res['duration']:.1f}초\n")

    print("=== 대사 (STT) ===")
    for t in res["transcript"]:
        print(f"[{t['start']:5.1f}s - {t['end']:5.1f}s] {t['text']}")
        if t.get("words"):
            print(
                f"    (단어 {len(t['words'])}개, 첫 3개: "
                + ", ".join(f"{w['word']}@{w['start']:.1f}s" for w in t["words"][:3])
                + ")"
            )

    print("\n=== 발화 구간 (VAD) ===")
    for s in res["speech"]:
        print(f"[{s['start']:5.1f}s - {s['end']:5.1f}s]")

    print("\n=== 침묵 구간 ===")
    for s in res["silence"]:
        print(
            f"[{s['start']:5.1f}s - {s['end']:5.1f}s] ({s['end'] - s['start']:.1f}초)"
        )
