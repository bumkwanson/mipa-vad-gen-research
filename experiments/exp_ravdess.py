"""
RAVDESS(정면 각도)에서 POSTER++ 표정 인식 정확도 측정
정답: 파일명의 감정 코드
목적: 각도가 통제된 조건에서의 정확도 → Seamless(하이앵글)와 비교
"""

import glob
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.poster_infer import PosterEmotion

# RAVDESS 감정 코드 -> 이름
RAVDESS_EMO = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happiness",
    "04": "Sadness",
    "05": "Anger",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise",
}
# POSTER++ 라벨 -> RAVDESS 이름 대응 (Calm은 POSTER++에 없음)
POSTER_MATCH = {
    "Neutral": "Neutral",
    "Happiness": "Happiness",
    "Sadness": "Sadness",
    "Anger": "Anger",
    "Fear": "Fear",
    "Disgust": "Disgust",
    "Surprise": "Surprise",
}

mp_face = mp.solutions.face_detection


def crop_face(frame, det, margin=0.2):
    h, w, _ = frame.shape
    bb = det.location_data.relative_bounding_box
    x = max(0, int((bb.xmin - bb.width * margin / 2) * w))
    y = max(0, int((bb.ymin - bb.height * margin / 2) * h))
    x2 = min(w, int((bb.xmin + bb.width * (1 + margin / 2)) * w))
    y2 = min(h, int((bb.ymin + bb.height * (1 + margin / 2)) * h))
    return frame[y:y2, x:x2] if (x2 > x and y2 > y) else None


def main():
    files = sorted(glob.glob("data/ravdess/Actor_01/01-01-*.mp4"))
    # calm(02) 제외 - POSTER++에 없음
    files = [f for f in files if f.split("/")[-1].split("-")[2] != "02"]
    print(f"대상 파일 {len(files)}개 (calm 제외)\n")

    emo = PosterEmotion()
    face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    confusion = defaultdict(lambda: defaultdict(int))
    correct = total = 0

    for path in files:
        code = path.split("/")[-1].split("-")[2]
        gt = RAVDESS_EMO[code]

        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 영상 중앙 부근 프레임들만 (감정이 뚜렷한 구간), 5프레임 간격
        votes = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if n * 0.3 < idx < n * 0.8 and idx % 5 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fres = face_det.process(rgb)
                if fres.detections:
                    best = max(fres.detections, key=lambda d: d.score[0])
                    fc = crop_face(frame, best)
                    if fc is not None and fc.size > 0:
                        pred = emo.predict(fc)["label"]
                        votes.append(POSTER_MATCH.get(pred, pred))
            idx += 1
        cap.release()

        if votes:
            # 클립 대표 예측 = 최빈값
            from collections import Counter

            pred_final = Counter(votes).most_common(1)[0][0]
            confusion[gt][pred_final] += 1
            correct += int(pred_final == gt)
            total += 1

    print("=== 클립 단위 정확도 (정면 각도, RAVDESS Actor_01) ===")
    accuracy = 100 * correct / total if total else 0.0
    print(f"정확도: {correct}/{total} ({accuracy:.1f}%)\n")

    print("=== 감정별 혼동 (정답 -> 예측 최빈) ===")
    for gt in sorted(confusion.keys()):
        tot = sum(confusion[gt].values())
        preds = ", ".join(
            f"{p}:{c}" for p, c in sorted(confusion[gt].items(), key=lambda x: -x[1])
        )
        acc = 100 * confusion[gt].get(gt, 0) / tot
        print(f"{gt:10s} ({acc:.0f}%): {preds}")


if __name__ == "__main__":
    main()
