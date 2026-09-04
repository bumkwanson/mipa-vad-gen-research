"""영상에서 프레임별 제스처 감지 결과를 시간순으로 출력"""
import sys
import cv2
sys.path.insert(0, 'src/vision')
from gesture import GestureDetector


def main():
    if len(sys.argv) < 2:
        print("사용법: python test_gesture_video.py <영상경로>")
        return

    cap = cv2.VideoCapture(sys.argv[1])
    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = GestureDetector()
    print(f"영상 fps: {fps:.1f}\n")

    frame_idx = 0
    # 동작이 바뀔 때만 출력 (매 프레임 다 찍으면 지저분)
    prev_gestures = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps

        res = detector.detect(frame)
        gestures = tuple(sorted(res['gestures']))

        if gestures != prev_gestures:
            print(f"[t={t:.1f}s] 고개:{res['head']}, 손:{res['hands_detected']}, 동작:{set(res['gestures'])}")
            prev_gestures = gestures

        frame_idx += 1

    detector.close()
    cap.release()
    print(f"\n총 {frame_idx}프레임 처리 완료")


if __name__ == '__main__':
    main()
