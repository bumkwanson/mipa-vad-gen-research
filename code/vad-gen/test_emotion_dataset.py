"""
7개 감정 영상에 대해 표정(POSTER++) + 제스처 통합 검증.
각 영상의 실제 라벨(폴더명)과 검출 결과를 비교.
"""
import sys
import os
import cv2
import numpy as np
from collections import Counter

sys.path.insert(0, 'src/vision')
import mediapipe as mp
from poster_infer import PosterEmotion
from gesture import GestureDetector

mp_face = mp.solutions.face_detection

EMOTION_DIR = os.path.expanduser('~/vad-gen/emotion_video/data/video_samples/emotions')
VIDEOS = {
    'Angry': 'Angry/Angry.mp4',
    'Disgust': 'Disgust/Disgust.mp4',
    'Fear': 'Fear/Fear.mov',
    'Happy': 'Happy/Happy.mp4',
    'Neutral': 'Neutral/Neutral.mp4',
    'Sad': 'Sad/Sad.mp4',
    'Surprised': 'Surprised/Surprised.mp4',
}

# POSTER++ 라벨 -> 폴더 라벨 매핑 (표기 통일)
LABEL_MAP = {
    'Anger': 'Angry', 'Disgust': 'Disgust', 'Fear': 'Fear',
    'Happiness': 'Happy', 'Neutral': 'Neutral', 'Sadness': 'Sad',
    'Surprise': 'Surprised',
}


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
    emotion_model = PosterEmotion()
    gesture_det = GestureDetector()
    face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    print("=" * 70)
    print(f"{'실제라벨':<10} {'표정검출(최빈)':<20} {'일치':<5} {'감지된 동작'}")
    print("=" * 70)

    correct = 0
    total = 0

    for true_label, relpath in VIDEOS.items():
        path = os.path.join(EMOTION_DIR, relpath)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"{true_label:<10} [영상 열기 실패]")
            continue

        emotion_votes = []
        gesture_set = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 표정: 얼굴 검출 -> crop -> POSTER++
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fres = face_det.process(rgb)
            if fres.detections:
                best = max(fres.detections, key=lambda d: d.score[0])
                fc = crop_face(frame, best)
                if fc is not None and fc.size > 0:
                    emo = emotion_model.predict(fc)
                    emotion_votes.append(emo['label'])

            # 제스처
            gres = gesture_det.detect(frame)
            gesture_set |= gres['gestures']

        cap.release()

        if emotion_votes:
            top_emo = Counter(emotion_votes).most_common(1)[0][0]
            mapped = LABEL_MAP.get(top_emo, top_emo)
            match = "O" if mapped == true_label else "X"
            if mapped == true_label:
                correct += 1
            total += 1
        else:
            top_emo = "없음"
            mapped = "-"
            match = "-"

        gest_str = ', '.join(sorted(gesture_set)) if gesture_set else '없음'
        print(f"{true_label:<10} {top_emo+' -> '+mapped:<20} {match:<5} {gest_str}")

    print("=" * 70)
    if total > 0:
        print(f"표정 정확도: {correct}/{total} ({100*correct/total:.1f}%)")

    gesture_det.close()
    face_det.close()


if __name__ == '__main__':
    main()
