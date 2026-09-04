"""vad-gen 전체 파이프라인 실행 진입점."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="영상에서 대사와 비언어 화면해설을 포함한 SRT를 생성합니다."
    )
    parser.add_argument("video", type=Path, help="분석할 영상 파일")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="출력 폴더 (기본값: outputs)",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="POSTER++ 가중치 폴더 (기본값: 프로젝트의 weights 또는 VADGEN_WEIGHTS_DIR)",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="추론 장치"
    )
    parser.add_argument(
        "--language", default="en", help="Whisper 언어 코드 (기본값: en)"
    )
    parser.add_argument(
        "--whisper-model", default="large-v3-turbo", help="faster-whisper 모델 이름"
    )
    parser.add_argument(
        "--ollama-model", default="exaone3.5:7.8b", help="Ollama 모델 이름"
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat"),
        help="Ollama chat API 주소",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Ollama 대신 감지된 고정 표현만 조합합니다.",
    )
    parser.add_argument(
        "--srt-only",
        action="store_true",
        help="SRT만 저장하고 자막 영상은 만들지 않습니다.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not video_path.is_file():
        print(f"오류: 영상 파일을 찾을 수 없습니다: {video_path}", file=sys.stderr)
        return 2

    try:
        from src.align.timeline import build_timeline
        from src.audio.audio_analysis import AudioAnalyzer
        from src.generate.describe import DescriptionGenerator, format_described
        from src.output.srt_export import burn_subtitles, save_srt
        from src.pipeline import VideoAnalyzer

        generator = DescriptionGenerator(
            model=args.ollama_model,
            base_url=args.ollama_url,
            use_llm=not args.deterministic,
        )
        generator.ensure_available()

        print("영상 분석 중...")
        video_segments = VideoAnalyzer(
            weights_dir=args.weights_dir,
            device=args.device,
        ).analyze(video_path)

        print("오디오 분석 중...")
        audio_result = AudioAnalyzer(
            whisper_model=args.whisper_model,
            device=args.device,
            language=args.language,
        ).analyze(video_path)

        print("타임라인 통합 중...")
        timeline = build_timeline(video_segments, audio_result)

        mode = "고정 표현" if args.deterministic else args.ollama_model
        print(f"화면해설 생성 중 ({mode})...")
        described = generator.generate(timeline)
        print()
        print(format_described(described))

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = video_path.stem
        srt_path = output_dir / f"{stem}.srt"
        save_srt(described, srt_path)
        print(f"SRT 저장: {srt_path}")

        if not args.srt_only:
            output_video = output_dir / f"{stem}_subtitled.mp4"
            print("자막 영상 생성 중 (ffmpeg)...")
            burn_subtitles(video_path, srt_path, output_video)
            print(f"자막 영상 저장: {output_video}")
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
