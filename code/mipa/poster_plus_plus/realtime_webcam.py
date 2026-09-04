"""
실시간 웹캠 기반 MIPA 전체 파이프라인
"""
import time
import cv2

from perception import MipaPerception
from fusion import FusionEngine, SILENCE_THRESHOLD_SEC, build_silence_context_prompt
from exaone_chat import ExaoneChat


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("웹캠을 열 수 없음 (인덱스 1). 다른 인덱스를 시도해보세요.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"웹캠 해상도: {width}x{height}")

    print("모델 로드 중 (EXAONE, POSTER++, MediaPipe)...")
    chat = ExaoneChat()
    perception = MipaPerception()
    engine = FusionEngine()
    engine.on_speech_ended()

    print("실시간 처리 시작. 'q' 키로 종료.\n")

    last_trigger_time = 0
    TRIGGER_COOLDOWN_SEC = 8.0

    last_response = ""

    frame_count = 0
    fps_timer = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없음")
                break

            now = time.time()
            result = perception.process_frame(frame)

            if not engine.is_user_speaking:
                engine.silence_buffer.add_frame(result, timestamp=now)
                silence_dur = engine.silence_buffer.current_silence_duration(now=now)

                if (silence_dur >= SILENCE_THRESHOLD_SEC and
                        now - last_trigger_time > TRIGGER_COOLDOWN_SEC):
                    summary = engine.silence_buffer.summarize()
                    context = build_silence_context_prompt(summary)
                    if context:
                        print(f"\n>>> [{time.strftime('%H:%M:%S')}] 침묵 트리거 발생!")
                        print(f">>> 컨텍스트: {context}")

                        gen_start = time.time()
                        response = chat.generate(context)
                        gen_time = time.time() - gen_start

                        print(f">>> EXAONE 응답 ({gen_time:.1f}초): {response}\n")
                        last_response = response
                        last_trigger_time = time.time()

            display = frame.copy()
            if result['face_detected']:
                emo = result['emotion']['label']
                conf = result['emotion']['confidence']
                cv2.putText(display, f"{emo} ({conf:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            silence_dur = engine.silence_buffer.current_silence_duration(now=now)
            cv2.putText(display, f"Silence: {silence_dur:.1f}s", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if last_response:
                cv2.putText(display, "EXAONE responded (see console)", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow('MIPA Full Pipeline (q to quit)', display)

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_timer
                actual_fps = 30 / elapsed
                print(f"[실시간 fps: {actual_fps:.1f}]")
                fps_timer = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        perception.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
