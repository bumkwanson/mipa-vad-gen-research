"""영상 -> 프레임별 분석 -> 스무딩 -> 구간화 결과 출력"""
import sys
import cv2
sys.path.insert(0, 'src/vision')
sys.path.insert(0, 'src/align')
import mediapipe as mp
from poster_infer import PosterEmotion
from gesture import GestureDetector
from segment import smooth_labels, smooth_gesture_sets, segment, format_segments

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
        print("사용법: python test_segment.py <영상경로>")
        return

    cap = cv2.VideoCapture(sys.argv[1])
    fps = cap.get(cv2.CAP_PROP_FPS)

    emotion_model = PosterEmotion()
    gesture_det = GestureDetector()
    face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    times, emotions, gestures = [], [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fres = face_det.process(rgb)
        emo_label = None
        if fres.detections:
            best = max(fres.detections, key=lambda d: d.score[0])
            fc = crop_face(frame, best)
            if fc is not None and fc.size > 0:
                emo_label = emotion_model.predict(fc)['label']

        gres = gesture_det.detect(frame)

        times.append(t)
        emotions.append(emo_label)
        gestures.append(gres['gestures'])
        frame_idx += 1

    gesture_det.close()
    face_det.close()
    cap.release()

    # 윈도우 크기: 약 0.4초 분량
    win = max(3, int(fps * 0.4) | 1)  # 홀수 보정
    print(f"프레임 {frame_idx}개, fps {fps:.1f}, 스무딩 윈도우 {win}프레임\n")

    sm_emo = smooth_labels(emotions, window=win)
    sm_gest = smooth_gesture_sets(gestures, window=win, ratio=0.5)
    segs = segment(times, sm_emo, sm_gest, min_duration=0.8)

    print("=== 스무딩 + 구간화 결과 ===")
    print(format_segments(segs))
    print(f"\n총 {len(segs)}개 구간")


if __name__ == '__main__':
    main()
