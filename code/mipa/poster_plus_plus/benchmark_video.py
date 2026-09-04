"""
영상 파일 기반 MIPA perception 파이프라인 벤치마크
웹캠이 없을 때 mp4 등 영상 파일로 fps/지연시간을 측정.

사용법:
    python benchmark_video.py <영상경로> [--every N] [--max-frames M]

--every N: N프레임마다 1번만 처리 (기본 1 = 매 프레임)
--max-frames M: 처음 M프레임만 처리하고 종료 (기본: 끝까지)
"""
import argparse
import time
import cv2
import numpy as np

from perception import MipaPerception


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('--every', type=int, default=1, help='N프레임마다 1번 처리')
    parser.add_argument('--max-frames', type=int, default=None, help='처리할 최대 프레임 수')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"영상을 열 수 없음: {args.video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"영상 정보: {src_fps:.1f} fps, 총 {total_frames} 프레임")

    perception = MipaPerception()
    print("Perception 모듈 로드 완료, 벤치마크 시작\n")

    frame_idx = 0
    processed_count = 0
    latencies = []
    face_detected_count = 0
    emotion_counts = {}
    profile_sums = {}

    bench_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if (frame_idx - 1) % args.every != 0:
            continue

        t0 = time.time()
        result = perception.process_frame(frame, profile=True)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        for k, v in result['_profile'].items():
            profile_sums[k] = profile_sums.get(k, 0) + v

        if result['face_detected']:
            face_detected_count += 1
            label = result['emotion']['label_kr']
            emotion_counts[label] = emotion_counts.get(label, 0) + 1

        processed_count += 1

        if processed_count % 30 == 0:
            avg_latency = sum(latencies[-30:]) / len(latencies[-30:])
            print(f"[{processed_count}프레임 처리] 최근 30프레임 평균 지연: {avg_latency:.1f}ms "
                  f"({1000/avg_latency:.1f} fps 처리 가능)")

        if args.max_frames and processed_count >= args.max_frames:
            break

    total_time = time.time() - bench_start
    perception.close()
    cap.release()

    print("\n" + "=" * 50)
    print("벤치마크 결과")
    print("=" * 50)
    print(f"처리한 프레임 수: {processed_count}")
    print(f"전체 소요 시간: {total_time:.2f}초")
    print(f"평균 지연시간: {np.mean(latencies):.1f}ms")
    print(f"최소/최대 지연시간: {np.min(latencies):.1f}ms / {np.max(latencies):.1f}ms")
    print(f"P95 지연시간: {np.percentile(latencies, 95):.1f}ms")
    print(f"처리 가능 fps: {1000 / np.mean(latencies):.1f}")
    print(f"얼굴 검출률: {face_detected_count}/{processed_count} "
          f"({100 * face_detected_count / processed_count:.1f}%)")
    print(f"감정 분포: {emotion_counts}")
    print("\n구간별 평균 소요시간:")
    for k, v in profile_sums.items():
        print(f"  {k}: {v / processed_count:.1f}ms")


if __name__ == '__main__':
    main()