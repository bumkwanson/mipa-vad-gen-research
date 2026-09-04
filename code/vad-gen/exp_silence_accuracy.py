"""
[핵심 실험] 발화 구간 vs 침묵 구간에서 POSTER++ 표정 인식 정확도 비교
정답: Seamless Interaction의 emotion_scores (Imitator 모델 출력)
가설: 침묵 구간에서 정확도가 더 높다 (발화 중 입 움직임이 표정 인식을 교란)
"""
import sys, glob, json
import numpy as np
import cv2
sys.path.insert(0, 'src/vision')
sys.path.insert(0, 'src/audio')
import mediapipe as mp
from poster_infer import PosterEmotion
from audio_analysis import AudioAnalyzer

# Seamless 8감정 -> POSTER++ 7감정 매핑 (Contempt는 POSTER++에 없음)
SEAMLESS = ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
# POSTER++ 라벨을 Seamless 인덱스에 대응
POSTER_TO_SEAMLESS = {
    'Anger': 0, 'Disgust': 2, 'Fear': 3, 'Happiness': 4,
    'Neutral': 5, 'Sadness': 6, 'Surprise': 7,
}

mp_face = mp.solutions.face_detection

def crop_face(frame, det, margin=0.2):
    h, w, _ = frame.shape
    bb = det.location_data.relative_bounding_box
    x = max(0, int((bb.xmin - bb.width*margin/2)*w))
    y = max(0, int((bb.ymin - bb.height*margin/2)*h))
    x2 = min(w, int((bb.xmin + bb.width*(1+margin/2))*w))
    y2 = min(h, int((bb.ymin + bb.height*(1+margin/2))*h))
    return frame[y:y2, x:x2] if (x2>x and y2>y) else None


def get_speech_frames(mp4_path, n_frames, fps):
    """VAD로 각 프레임이 발화(True)인지 침묵(False)인지"""
    analyzer = AudioAnalyzer()
    res = analyzer.analyze(mp4_path)
    is_speech = np.zeros(n_frames, dtype=bool)
    for seg in res['speech']:
        s = int(seg['start']*fps)
        e = int(seg['end']*fps)
        is_speech[s:min(e,n_frames)] = True
    return is_speech


def run_file(target, npz_files, sample_every=5):
    npz = [f for f in npz_files if target in f][0]
    mp4 = npz.replace('.npz','.mp4')
    data = np.load(npz)
    scores = data['movement:emotion_scores']
    valid = data['movement:is_valid']
    nonzero = scores.sum(axis=1) > 1e-6
    gt_valid = valid & nonzero
    gt_labels = np.argmax(scores, axis=1)

    cap = cv2.VideoCapture(mp4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[{target}] VAD 분석 중...")
    is_speech = get_speech_frames(mp4, n_frames, fps)

    emo = PosterEmotion()
    face_det = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # 결과 집계: (구간종류, 정답 여부)
    speech_correct = speech_total = 0
    silence_correct = silence_total = 0

    print(f"[{target}] 표정 인식 중 (매 {sample_every}프레임)...")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0 and idx < len(gt_valid) and gt_valid[idx]:
            gt = gt_labels[idx]
            if gt != 1:  # Contempt 제외
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fres = face_det.process(rgb)
                if fres.detections:
                    best = max(fres.detections, key=lambda d: d.score[0])
                    fc = crop_face(frame, best)
                    if fc is not None and fc.size > 0:
                        pred = emo.predict(fc)['label']
                        pred_seamless = POSTER_TO_SEAMLESS.get(pred, -1)
                        correct = (pred_seamless == gt)
                        if is_speech[idx]:
                            speech_total += 1
                            speech_correct += int(correct)
                        else:
                            silence_total += 1
                            silence_correct += int(correct)
        idx += 1

    cap.release()
    emo_acc_speech = 100*speech_correct/speech_total if speech_total else 0
    emo_acc_silence = 100*silence_correct/silence_total if silence_total else 0
    print(f"[{target}] 발화: {speech_correct}/{speech_total} ({emo_acc_speech:.1f}%), "
          f"침묵: {silence_correct}/{silence_total} ({emo_acc_silence:.1f}%)")
    return (speech_correct, speech_total, silence_correct, silence_total)


def main():
    npz_files = sorted(glob.glob('data/seamless/**/*.npz', recursive=True))
    targets = ['P0844A','P1277A','P0852','P0947']

    tot_sc=tot_st=tot_zc=tot_zt=0
    for t in targets:
        sc, st, zc, zt = run_file(t, npz_files)
        tot_sc+=sc; tot_st+=st; tot_zc+=zc; tot_zt+=zt

    print("\n" + "="*50)
    print("=== 전체 종합 ===")
    print(f"발화 구간 정확도: {tot_sc}/{tot_st} ({100*tot_sc/tot_st:.1f}%)")
    print(f"침묵 구간 정확도: {tot_zc}/{tot_zt} ({100*tot_zc/tot_zt:.1f}%)")
    print("="*50)


if __name__ == '__main__':
    main()
