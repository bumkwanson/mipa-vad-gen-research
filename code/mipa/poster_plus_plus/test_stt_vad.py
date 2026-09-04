"""
faster-whisper STT + silero-VAD 검증 스크립트
마이크 없이 오디오 파일로 파이프라인 동작을 확인.

사용법:
    python test_stt_vad.py <오디오파일.wav>
"""
import sys
import time


def test_whisper(audio_path):
    print("=" * 50)
    print("1. faster-whisper STT 테스트")
    print("=" * 50)

    from faster_whisper import WhisperModel

    t0 = time.time()
    # large-v3-turbo: Jarvis 프로젝트에서 사용한 모델, 속도/정확도 균형이 좋음
    # compute_type="float16": GPU에서 속도를 위해 절반 정밀도 사용
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    print(f"모델 로드 시간: {time.time() - t0:.1f}초\n")

    t0 = time.time()
    segments, info = model.transcribe(audio_path, language="ko", beam_size=5)
    segments = list(segments)  # generator이므로 리스트로 변환해야 실제 처리시간 측정됨
    transcribe_time = time.time() - t0

    print(f"감지된 언어: {info.language} (확률: {info.language_probability:.2f})")
    print(f"오디오 길이: {info.duration:.1f}초")
    print(f"STT 처리 시간: {transcribe_time:.1f}초 (실시간 대비 {info.duration/transcribe_time:.1f}배속)\n")

    print("전사 결과:")
    full_text = ""
    for seg in segments:
        print(f"  [{seg.start:.1f}s -> {seg.end:.1f}s] {seg.text}")
        full_text += seg.text

    return full_text


def test_vad(audio_path):
    print("\n" + "=" * 50)
    print("2. silero-VAD 발화/침묵 구간 테스트")
    print("=" * 50)

    import torch
    import numpy as np
    import soundfile as sf

    torch.set_num_threads(1)

    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (get_speech_timestamps, _, _, *_) = utils

    # silero-vad의 read_audio()는 최신 torchaudio(2.9+)에서 torchcodec을 요구해
    # 버전 충돌이 남. soundfile로 직접 읽어서 우회 (더 가볍고 의존성 적음).
    wav_np, sr = sf.read(audio_path, dtype='float32')
    if wav_np.ndim > 1:  # 혹시 스테레오로 남아있으면 모노로 변환
        wav_np = wav_np.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"샘플레이트가 16000Hz가 아님: {sr}Hz (ffmpeg에서 -ar 16000으로 변환 필요)")
    wav = torch.from_numpy(wav_np)

    # Jarvis 노트 기준 파라미터
    speech_timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=16000,
        threshold=0.7,  # Jarvis에서 검증된 값 (0.5는 너무 민감)
    )

    print(f"감지된 발화 구간 수: {len(speech_timestamps)}\n")
    for i, ts in enumerate(speech_timestamps):
        start_sec = ts['start'] / 16000
        end_sec = ts['end'] / 16000
        print(f"  발화 {i+1}: {start_sec:.1f}s ~ {end_sec:.1f}s (길이 {end_sec-start_sec:.1f}s)")

    # 침묵 구간 계산 (발화 구간 사이사이)
    if len(speech_timestamps) > 1:
        print("\n침묵 구간:")
        for i in range(len(speech_timestamps) - 1):
            silence_start = speech_timestamps[i]['end'] / 16000
            silence_end = speech_timestamps[i+1]['start'] / 16000
            silence_dur = silence_end - silence_start
            if silence_dur > 0.1:  # 너무 짧은 건 노이즈로 무시
                print(f"  침묵: {silence_start:.1f}s ~ {silence_end:.1f}s (길이 {silence_dur:.1f}s)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python test_stt_vad.py <오디오파일.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]

    text = test_whisper(audio_path)
    test_vad(audio_path)

    print("\n" + "=" * 50)
    print("전체 테스트 완료")
    print("=" * 50)