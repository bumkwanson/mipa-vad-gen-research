"""
perception.py + fusion.py 통합 테스트
영상 파일을 프레임 단위로 흘리면서, 영상 자체의 타임스탬프(frame_idx/fps) 기준으로
침묵 버퍼가 쌓이고 트리거되는지 확인.

주의: 이 테스트는 VAD 없이 "영상 전체를 침묵 상태로 가정"하고 돌린다.
실제 발화/침묵 전환은 다음 단계(STT/VAD 연동)에서 다룸.

사용법:
    python test_fusion_integration.py <영상경로>
"""
import sys
import cv2

from perception import MipaPerception
from fusion import FusionEngine, SILENCE_THRESHOLD_SEC


def main():
    if len(sys.argv) < 2:
        print("사용법: python test_fusion_integration.py <영상경로>")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"영상을 열 수 없음: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"영상 fps: {fps:.1f}\n")

    perception = MipaPerception()
    engine = FusionEngine()

    # 영상 시작 시점을 침묵 시작으로 간주 (VAD 미연동 상태 가정)
    engine.on_speech_ended()

    frame_idx = 0
    triggered_once = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 영상 자체의 타임스탬프를 사용 (실제 처리 속도와 무관하게 일정한 흐름 보장)
        video_timestamp = frame_idx / fps

        result = perception.process_frame(frame)

        # fusion의 add_frame이 내부적으로 time.time()을 쓰지 않도록
        # 영상 타임스탬프를 명시적으로 전달
        if not engine.is_user_speaking:
            engine.silence_buffer.add_frame(result, timestamp=video_timestamp)

            silence_dur = engine.silence_buffer.current_silence_duration(now=video_timestamp)

            if frame_idx % 30 == 0:
                emo = result['emotion']['label_kr'] if result['emotion'] else '없음'
                print(f"[t={video_timestamp:.1f}s] 감정: {emo}, "
                      f"누적 침묵: {silence_dur:.1f}s")

            if silence_dur >= SILENCE_THRESHOLD_SEC and not triggered_once:
                summary = engine.silence_buffer.summarize()
                from fusion import build_silence_context_prompt
                context = build_silence_context_prompt(summary)
                if context:
                    print(f"\n>>> [t={video_timestamp:.1f}s] 침묵 트리거 발생!")
                    print(f">>> {context}\n")
                    triggered_once = True  # 데모 목적으로 1회만 출력

        frame_idx += 1

    perception.close()
    cap.release()

    if not triggered_once:
        print("\n영상 전체에서 침묵 트리거 조건이 충족되지 않음 "
              "(연속 부정 감정이 부족했거나 영상이 너무 짧음)")


if __name__ == '__main__':
    main()