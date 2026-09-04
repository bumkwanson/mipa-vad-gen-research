"""영상에서 몇 프레임 뽑아 얼굴 검출 박스 + 표정을 이미지로 저장"""
import sys, cv2, os
sys.path.insert(0, 'src/vision')
import mediapipe as mp
from poster_infer import PosterEmotion

mp_face = mp.solutions.face_detection

def crop_face(frame, det, margin=0.2):
    h, w, _ = frame.shape
    bb = det.location_data.relative_bounding_box
    x = max(0, int((bb.xmin - bb.width*margin/2)*w))
    y = max(0, int((bb.ymin - bb.height*margin/2)*h))
    x2 = min(w, int((bb.xmin + bb.width*(1+margin/2))*w))
    y2 = min(h, int((bb.ymin + bb.height*(1+margin/2))*h))
    return (x, y, x2, y2)

def main():
    path = sys.argv[1]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    emo = PosterEmotion()
    face_det = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    os.makedirs('outputs/face_check', exist_ok=True)

    # 2초 간격으로 프레임 뽑기
    for t in range(0, int(total/fps), 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t*fps))
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fres = face_det.process(rgb)
        if fres.detections:
            best = max(fres.detections, key=lambda d: d.score[0])
            x, y, x2, y2 = crop_face(frame, best)
            fc = frame[y:y2, x:x2]
            label = "?"
            if fc.size > 0:
                e = emo.predict(fc)
                label = f"{e['label']} {e['confidence']:.2f}"
            cv2.rectangle(frame, (x,y), (x2,y2), (0,255,0), 3)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            print(f"t={t}s: 얼굴검출 O, {label}, 박스=({x},{y})-({x2},{y2})")
        else:
            print(f"t={t}s: 얼굴검출 X")
        # 축소해서 저장 (세로영상이라 큼)
        small = cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3))
        cv2.imwrite(f'outputs/face_check/t{t:02d}.jpg', small)

    face_det.close()
    cap.release()
    print("\n저장됨: outputs/face_check/")

if __name__ == '__main__':
    main()
