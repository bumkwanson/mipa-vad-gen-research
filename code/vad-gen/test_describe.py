"""영상 -> 전체 분석 -> LLM 화면해설 생성"""
import sys
sys.path.insert(0, 'src/vision')
sys.path.insert(0, 'src/audio')
sys.path.insert(0, 'src/align')
sys.path.insert(0, 'src/generate')

from test_timeline import analyze_video
from audio_analysis import AudioAnalyzer
from timeline import build_timeline
from describe import DescriptionGenerator, format_described


def main():
    if len(sys.argv) < 2:
        print("사용법: python test_describe.py <영상경로>")
        return
    path = sys.argv[1]

    print("영상 분석 중...")
    video_segs = analyze_video(path)
    print("오디오 분석 중...")
    audio_res = AudioAnalyzer().analyze(path)
    tl = build_timeline(video_segs, audio_res)


    print("\n화면해설 생성 중...")
    gen = DescriptionGenerator()
    described = gen.generate(tl)

    if described:
        print("\n=== 생성된 화면해설 ===")
        print(format_described(described))


if __name__ == '__main__':
    main()
