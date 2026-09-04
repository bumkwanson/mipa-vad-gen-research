"""
영상에서 표정(POSTER++) + 제스처를 함께 시간순으로 출력.
상태가 바뀔 때만 출력.
"""
import sys
import cv2
sys.path.insert(0, 'src/vision')
import mediapipe as mp
from poster_infer import PosterEmotion
from gesture import GestureDetector

mp_face = mp.solutions.face_detection


def crop_face(frame, detection, margin=0.2):
    h, w, _ = frame.shape
    bb = detection.location_data.relative_bounding_box
    x = max(0, int((bb.xmin - bb.width*margin/2)*w))
    y = max(0, int((bb.ymin - bb.height*margin/2)*h))
    x2 = min(w, int((bb.xmin + bb.width*(1+margin/2))*w))
    y2 = min(h, int((bb.ymin + bb.height*(1+margin/2))*h))
    if x2 <= x or y2 <= y:
        return None
    return frame[y:y2, x:x2]


def main():
    if len(sys.argv) < 2:
        print("사용법: python test_full_video.py <영상경로>")
        return

    cap = cv2.VideoCapture(sys.argv[1])
    fps = cap.get(cv2.CAP_PROP_FPS)

    emotion_model = PosterEmotion()
    gesture_det = GestureDetector()
    face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    print(f"영상 fps: {fps:.1f}\n")
    print(f"{'시간':<8} {'표정':<12} {'conf':<6} {'동작'}")
    print("-" * 60)

    frame_idx = 0
    prev_state = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps

        # 표정
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fres = face_det.process(rgb)
        emo_label, emo_conf = "없음", 0.0
        if fres.detections:
            best = max(fres.detections, key=lambda d: d.score[0])
            fc = crop_face(frame, best)
            if fc is not None and fc.size > 0:
                e = emotion_model.predict(fc)
                emo_label, emo_conf = e['label'], e['confidence']

        # 제스처
        gres = gesture_det.detect(frame)
        gest = tuple(sorted(gres['gestures']))

        state = (emo_label, gest)
        if state != prev_state:
            gstr = ', '.join(gest) if gest else '-'
            print(f"{t:>6.1f}s  {emo_label:<12} {emo_conf:<6.2f} {gstr}")
            prev_state = state

        frame_idx += 1

    gesture_det.close()
    face_det.close()
    cap.release()
    print(f"\n총 {frame_idx}프레임 처리 완료")


if __name__ == '__main__':
    main()
