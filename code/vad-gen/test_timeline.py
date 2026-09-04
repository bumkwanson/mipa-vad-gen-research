"""영상 -> 영상분석 + 오디오분석 -> 시간축 통합 결과 출력"""
import sys
import cv2
sys.path.insert(0, 'src/vision')
sys.path.insert(0, 'src/audio')
sys.path.insert(0, 'src/align')

import mediapipe as mp
from poster_infer import PosterEmotion
from gesture import GestureDetector
from segment import smooth_labels, smooth_gesture_sets, segment
from audio_analysis import AudioAnalyzer
from timeline import build_timeline, format_timeline

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


def analyze_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    emotion_model = PosterEmotion()
    gesture_det = GestureDetector()
    face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    times, emotions, gestures = [], [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fres = face_det.process(rgb)
        emo = None
        if fres.detections:
            best = max(fres.detections, key=lambda d: d.score[0])
            fc = crop_face(frame, best)
            if fc is not None and fc.size > 0:
                emo = emotion_model.predict(fc)['label']
        gres = gesture_det.detect(frame)
        times.append(t); emotions.append(emo); gestures.append(gres['gestures'])
        idx += 1

    gesture_det.close(); face_det.close(); cap.release()

    win = max(3, int(fps * 0.4) | 1)
    sm_emo = smooth_labels(emotions, window=win)
    sm_gest = smooth_gesture_sets(gestures, window=win, ratio=0.5)
    return segment(times, sm_emo, sm_gest, min_duration=0.8)


def main():
    if len(sys.argv) < 2:
        print("사용법: python test_timeline.py <영상경로>")
        return
    path = sys.argv[1]

    print("영상 분석 중...")
    video_segs = analyze_video(path)

    print("오디오 분석 중...")
    analyzer = AudioAnalyzer()
    audio_res = analyzer.analyze(path)

    print("\n=== 통합 타임라인 ===\n")
    tl = build_timeline(video_segs, audio_res)
    print(format_timeline(tl))


if __name__ == '__main__':
    main()
