"""영상 -> 전체 분석 -> SRT -> 자막 입힌 데모 영상"""
import sys
import os
sys.path.insert(0, 'src/vision')
sys.path.insert(0, 'src/audio')
sys.path.insert(0, 'src/align')
sys.path.insert(0, 'src/generate')
sys.path.insert(0, 'src/output')

from test_timeline import analyze_video
from audio_analysis import AudioAnalyzer
from timeline import build_timeline
from describe import DescriptionGenerator, format_described
from srt_export import save_srt, burn_subtitles


def main():
    if len(sys.argv) < 2:
        print("사용법: python test_export.py <영상경로>")
        return
    path = sys.argv[1]
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs('outputs', exist_ok=True)

    print("영상 분석 중...")
    video_segs = analyze_video(path)

    print("오디오 분석 중...")
    analyzer = AudioAnalyzer()
    audio_res = analyzer.analyze(path)

    print("타임라인 통합 중...")
    tl = build_timeline(video_segs, audio_res)

    print("변화 기반 화면해설 생성 중...")
    described = DescriptionGenerator().generate(tl)
    print()
    print(format_described(described))

    srt_path = os.path.abspath(f'outputs/{base}.srt')
    save_srt(described, srt_path)
    print(f"SRT 저장: {srt_path}")

    out_video = os.path.abspath(f'outputs/{base}_subtitled.mp4')
    print("자막 입히는 중 (ffmpeg)...")
    result = burn_subtitles(os.path.abspath(path), srt_path, out_video)
    if result:
        print(f"데모 영상 저장: {out_video}")


if __name__ == '__main__':
    main()
