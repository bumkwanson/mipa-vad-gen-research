"""
MIPA Perception 모듈
- MediaPipe Face Detection: 얼굴 crop -> POSTER++ 입력
- MediaPipe Pose + Hands: 고개/몸 방향 + 손 랜드마크 (Holistic 대신 개별 모델 사용 - 더 가벼움)
- POSTER++: crop된 얼굴 -> 7class 감정 확률

한 프레임을 넣으면 감정 + pose/hands 신호를 한 번에 반환.
"""
import cv2
import numpy as np
import mediapipe as mp

from inference import PosterEmotionRecognizer

mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


class MipaPerception:
    def __init__(self, poster_checkpoint='./checkpoint/raf-db-model_best.pth', device='cuda'):
        self.emotion_model = PosterEmotionRecognizer(poster_checkpoint, device=device)

        # min_detection_confidence: 너무 낮으면 오검출, 너무 높으면 놓침. 0.5가 기본값이자 무난한 지점.
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=0,  # 0: 2m 이내 근거리(웹캠 앞 사용자에 적합), 1: 원거리
            min_detection_confidence=0.5
        )

        # Holistic 대신 Pose+Hands를 개별로 사용: Holistic은 내부적으로 얼굴 468-landmark도
        # 항상 같이 계산하는데, 표정은 POSTER++가 전담하니 그 계산이 낭비였음.
        # 필요한 것(pose, hands)만 쓰면 그만큼 지연시간이 줄어듦.
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # 0(가장 가벼움)~2(가장 정확) 중 실시간 처리엔 1이 균형점
        )
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _crop_face(self, frame_bgr, detection, margin=0.2):
        """MediaPipe 검출 박스를 기준으로 margin만큼 여유를 두고 crop"""
        h, w, _ = frame_bgr.shape
        bbox = detection.location_data.relative_bounding_box

        x = max(0, int((bbox.xmin - bbox.width * margin / 2) * w))
        y = max(0, int((bbox.ymin - bbox.height * margin / 2) * h))
        x2 = min(w, int((bbox.xmin + bbox.width * (1 + margin / 2)) * w))
        y2 = min(h, int((bbox.ymin + bbox.height * (1 + margin / 2)) * h))

        if x2 <= x or y2 <= y:
            return None
        return frame_bgr[y:y2, x:x2]

    def process_frame(self, frame_bgr, profile=False):
        """
        frame_bgr: 웹캠에서 받은 원본 BGR 프레임
        profile: True면 각 단계별 소요시간(ms)을 result['_profile']에 담아 반환
        반환: dict
            face_detected: bool
            emotion: {'label', 'label_kr', 'confidence', 'probs'} or None
            pose_landmarks: pose 랜드마크 (고개 방향 추정용) or None
            hands_detected: {'left': bool, 'right': bool}
        """
        import time
        t = {}

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # MediaPipe 권장: 처리 중 원본 불변으로 표시해 속도 향상

        t0 = time.time()
        face_results = self.face_detector.process(frame_rgb)
        t['face_detection'] = (time.time() - t0) * 1000

        t0 = time.time()
        pose_results = self.pose.process(frame_rgb)
        t['pose'] = (time.time() - t0) * 1000

        t0 = time.time()
        hands_results = self.hands.process(frame_rgb)
        t['hands'] = (time.time() - t0) * 1000

        result = {
            'face_detected': False,
            'emotion': None,
            'pose_landmarks': None,
            'hands_detected': {'left': False, 'right': False},
        }

        t0 = time.time()
        if face_results.detections:
            # 가장 confidence 높은 얼굴 하나만 사용 (다인 상황은 아직 미지원)
            best_detection = max(
                face_results.detections,
                key=lambda d: d.score[0]
            )
            face_crop = self._crop_face(frame_bgr, best_detection)

            if face_crop is not None and face_crop.size > 0:
                result['face_detected'] = True
                result['emotion'] = self.emotion_model.predict(face_crop)
        t['poster_emotion'] = (time.time() - t0) * 1000

        if pose_results.pose_landmarks:
            result['pose_landmarks'] = pose_results.pose_landmarks

        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for handedness in hands_results.multi_handedness:
                # MediaPipe Hands는 좌우를 'Left'/'Right' 문자열로 반환
                # (참고: 카메라가 셀카처럼 좌우 반전 없이 촬영했다면 실제 신체 기준과
                #  화면상 좌우가 다를 수 있음 - 필요시 미러링 보정 추가)
                label = handedness.classification[0].label.lower()
                if label in result['hands_detected']:
                    result['hands_detected'][label] = True

        if profile:
            result['_profile'] = t

        return result

    def close(self):
        self.face_detector.close()
        self.pose.close()
        self.hands.close()


if __name__ == '__main__':
    import sys

    perception = MipaPerception()
    print("Perception 모듈 로드 완료 (POSTER++ + MediaPipe)")

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"이미지를 읽을 수 없음: {sys.argv[1]}")
            sys.exit(1)
        result = perception.process_frame(img)
        print(f"얼굴 검출: {result['face_detected']}")
        print(f"감정: {result['emotion']}")
        print(f"손 검출: {result['hands_detected']}")
    else:
        print("사용법: python perception.py <이미지경로>")

    perception.close()