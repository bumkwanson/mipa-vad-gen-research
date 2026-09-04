"""
제스처(동작) 감지 모듈 - vad-gen
MediaPipe pose + hands 랜드마크로부터 '관찰 가능한 동작'만 추출.
감정/의도 해석은 하지 않음 (화면해설 원칙: describe, not interpret).

출력 라벨은 모두 중립적 동작 서술:
    hand_near_face   : 손이 얼굴/입 근처
    hand_near_neck   : 손이 목/턱 근처
    head_turned      : 고개가 좌우로 돌아감
    head_down        : 고개를 숙임
    hands_lowered    : 손이 몸통 아래 (특별한 동작 없음)

감정 해석, 완화 표현 등은 이후 LLM 단계에서 담당.
"""
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# MediaPipe Pose 랜드마크 인덱스 (주요 지점)
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def _dist(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class GestureDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _pose_points(self, pose_landmarks, w, h):
        """필요한 pose 랜드마크를 (x, y) 픽셀 좌표 dict로"""
        lm = pose_landmarks.landmark
        pts = {}
        for idx in [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
                    MOUTH_LEFT, MOUTH_RIGHT, LEFT_SHOULDER, RIGHT_SHOULDER]:
            pts[idx] = (lm[idx].x * w, lm[idx].y * h)
        return pts

    def _hand_points(self, hand_landmarks, w, h):
        """손 랜드마크 전체를 픽셀 좌표 리스트로 (대표점: 손목 0, 중지끝 12)"""
        lm = hand_landmarks.landmark
        return {
            'wrist': (lm[0].x * w, lm[0].y * h),
            'index_tip': (lm[8].x * w, lm[8].y * h),
            'middle_tip': (lm[12].x * w, lm[12].y * h),
        }

    def detect(self, frame_bgr):
        """
        frame_bgr: 원본 프레임
        반환: dict {
            'gestures': set of str,   # 감지된 동작 라벨들
            'head': str,              # 'front' | 'turned' | 'down' | None
            'pose_detected': bool,
            'hands_detected': int,    # 감지된 손 개수
        }
        """
        h, w, _ = frame_bgr.shape
        rgb = frame_bgr[:, :, ::-1]  # BGR->RGB (copy 없이 뷰)

        pose_res = self.pose.process(rgb)
        hands_res = self.hands.process(rgb)

        result = {
            'gestures': set(),
            'head': None,
            'pose_detected': False,
            'hands_detected': 0,
        }

        if not pose_res.pose_landmarks:
            return result

        result['pose_detected'] = True
        pts = self._pose_points(pose_res.pose_landmarks, w, h)

        # 어깨 너비를 기준 스케일로 사용 (거리 임계값을 사람 크기에 정규화)
        shoulder_width = _dist(pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER])
        if shoulder_width < 1e-3:
            shoulder_width = w * 0.3  # fallback

        # --- 고개 방향 판정 (관찰 가능한 기하학적 사실만) ---
        # 코가 양 어깨 중심에서 좌우로 크게 벗어나면 고개 돌림
        shoulder_mid_x = (pts[LEFT_SHOULDER][0] + pts[RIGHT_SHOULDER][0]) / 2
        nose_offset_x = abs(pts[NOSE][0] - shoulder_mid_x)

        # 코가 어깨선보다 많이 아래로 내려오면 고개 숙임
        shoulder_mid_y = (pts[LEFT_SHOULDER][1] + pts[RIGHT_SHOULDER][1]) / 2
        nose_to_shoulder_y = pts[NOSE][1] - shoulder_mid_y  # 음수면 코가 위(정상)

        if nose_offset_x > shoulder_width * 0.35:
            result['head'] = 'turned'
            result['gestures'].add('head_turned')
        elif nose_to_shoulder_y > -shoulder_width * 0.3:
            # 코가 어깨에 가깝거나 아래 = 고개 숙임
            result['head'] = 'down'
            result['gestures'].add('head_down')
        else:
            result['head'] = 'front'

        # --- 손 위치 판정 ---
        face_center = pts[NOSE]
        neck_y = shoulder_mid_y  # 목/턱 근처 기준선
        chin_approx = ((pts[MOUTH_LEFT][0] + pts[MOUTH_RIGHT][0]) / 2,
                       (pts[MOUTH_LEFT][1] + pts[MOUTH_RIGHT][1]) / 2)

        if hands_res.multi_hand_landmarks:
            result['hands_detected'] = len(hands_res.multi_hand_landmarks)
            for hand_lm in hands_res.multi_hand_landmarks:
                hp = self._hand_points(hand_lm, w, h)
                # 손 대표점: 손가락 끝 평균
                hand_pt = ((hp['index_tip'][0] + hp['middle_tip'][0]) / 2,
                           (hp['index_tip'][1] + hp['middle_tip'][1]) / 2)

                # 손이 얼굴(코/입) 근처?
                if _dist(hand_pt, face_center) < shoulder_width * 0.5 or \
                   _dist(hand_pt, chin_approx) < shoulder_width * 0.4:
                    result['gestures'].add('hand_near_face')
                # 손이 목/턱~어깨 사이?
                elif abs(hand_pt[1] - neck_y) < shoulder_width * 0.4 and \
                        abs(hand_pt[0] - shoulder_mid_x) < shoulder_width * 0.6:
                    result['gestures'].add('hand_near_neck')

        # 손이 감지 안 됐거나 위쪽 동작이 없으면 기본 상태
        if not result['gestures'] or result['gestures'] == {'head_turned'} or \
           result['gestures'] == {'head_down'}:
            if result['hands_detected'] == 0:
                result['gestures'].add('hands_lowered')

        return result

    def close(self):
        self.pose.close()
        self.hands.close()


if __name__ == '__main__':
    import sys
    import cv2

    detector = GestureDetector()
    print("제스처 감지기 로드 완료")

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"이미지를 읽을 수 없음: {sys.argv[1]}")
            sys.exit(1)
        res = detector.detect(img)
        print(f"pose 감지: {res['pose_detected']}")
        print(f"손 개수: {res['hands_detected']}")
        print(f"고개 방향: {res['head']}")
        print(f"감지된 동작: {res['gestures']}")
    else:
        print("사용법: python gesture.py <이미지경로>")

    detector.close()
