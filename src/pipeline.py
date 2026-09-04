"""영상에서 표정·제스처 구간을 추출하는 파이프라인."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp

from .align.segment import segment, smooth_gesture_sets, smooth_labels
from .vision.gesture import GestureDetector
from .vision.poster_infer import PosterEmotion

mp_face = mp.solutions.face_detection


def crop_face(frame, detection, margin: float = 0.2):
    """MediaPipe 얼굴 경계 상자를 확장해 BGR 얼굴 이미지를 반환합니다."""
    height, width, _ = frame.shape
    box = detection.location_data.relative_bounding_box
    x1 = max(0, int((box.xmin - box.width * margin / 2) * width))
    y1 = max(0, int((box.ymin - box.height * margin / 2) * height))
    x2 = min(width, int((box.xmin + box.width * (1 + margin / 2)) * width))
    y2 = min(height, int((box.ymin + box.height * (1 + margin / 2)) * height))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


class VideoAnalyzer:
    def __init__(
        self,
        weights_dir: str | Path | None = None,
        device: str = "auto",
        face_margin: float = 0.2,
        min_segment_duration: float = 0.8,
    ) -> None:
        self.weights_dir = weights_dir
        self.device = device
        self.face_margin = face_margin
        self.min_segment_duration = min_segment_duration

    def analyze(self, video_path: str | Path):
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            capture.release()
            raise RuntimeError(f"영상 FPS를 읽을 수 없습니다: {path}")

        emotion_model = PosterEmotion(weights_dir=self.weights_dir, device=self.device)
        gesture_detector = GestureDetector()
        face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

        times: list[float] = []
        emotions: list[str | None] = []
        gestures: list[set[str]] = []

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_result = face_detector.process(rgb)
                emotion = None
                if face_result.detections:
                    best = max(face_result.detections, key=lambda item: item.score[0])
                    face = crop_face(frame, best, margin=self.face_margin)
                    if face is not None and face.size > 0:
                        emotion = emotion_model.predict(face)["label"]

                gesture_result = gesture_detector.detect(frame)
                times.append(frame_index / fps)
                emotions.append(emotion)
                gestures.append(gesture_result["gestures"])
                frame_index += 1
        finally:
            gesture_detector.close()
            face_detector.close()
            capture.release()

        if not times:
            raise RuntimeError(f"영상에서 프레임을 읽지 못했습니다: {path}")

        window = max(3, int(fps * 0.4) | 1)
        smoothed_emotions = smooth_labels(emotions, window=window)
        smoothed_gestures = smooth_gesture_sets(gestures, window=window, ratio=0.5)
        return segment(
            times,
            smoothed_emotions,
            smoothed_gestures,
            min_duration=self.min_segment_duration,
        )


def analyze_video(video_path: str | Path, **kwargs):
    """기존 호출부를 위한 간단한 함수형 인터페이스."""
    return VideoAnalyzer(**kwargs).analyze(video_path)
