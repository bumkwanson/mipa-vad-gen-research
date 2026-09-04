"""
MIPA Fusion 레이어
- 발화 중: STT 텍스트 + 순간 감정을 즉시 EXAONE 컨텍스트로
- 침묵 중: 감정/pose 신호를 sliding window 버퍼에 축적
         -> 침묵이 일정 시간(SILENCE_THRESHOLD) 이상 지속되면 버퍼를 요약해서
            "침묵 자체"를 LLM 컨텍스트로 전달 (MIPA 핵심 novelty)

Jarvis 프로젝트에서 검증된 파라미터를 기반으로 하되, 단발성 감정 판단이 아니라
침묵 구간 전체의 감정 궤적(trajectory)을 넘긴다는 점이 다름.
"""
import time
from collections import deque


# Jarvis 노트에서 확인된 값들
TRIGGER_EMOTIONS = {'슬픔', '두려움', '화남', '혐오'}
EMOTION_CONFIDENCE_THRESHOLD = 0.5
SILENCE_THRESHOLD_SEC = 1.5          # 이 이상 조용하면 "의미있는 침묵"으로 간주
MIN_CONSECUTIVE_TRIGGERS = 2          # 일시적 표정 변화 무시하기 위한 연속 감지 요구치

# sliding window 버퍼 최대 길이 (초 단위) - 너무 길게 쌓아두면 오래된 신호가
# 최근 상태를 오염시킬 수 있어 상한을 둠
MAX_BUFFER_SEC = 10.0


class SilenceBuffer:
    """
    침묵 구간 동안의 (timestamp, emotion, pose 존재여부) 프레임을 누적하는 sliding window.
    발화가 감지되면 flush()로 비우고, 침묵이 SILENCE_THRESHOLD를 넘기면
    summarize()로 LLM에 넘길 요약을 만든다.
    """

    def __init__(self):
        self.frames = deque()  # 각 원소: {'t': float, 'emotion': str, 'confidence': float}
        self.silence_start_t = None

    def add_frame(self, perception_result, timestamp=None):
        """perception.process_frame()의 결과를 받아 버퍼에 추가"""
        if timestamp is None:
            timestamp = time.time()

        if self.silence_start_t is None:
            self.silence_start_t = timestamp

        if perception_result.get('emotion'):
            self.frames.append({
                't': timestamp,
                'emotion': perception_result['emotion']['label_kr'],
                'confidence': perception_result['emotion']['confidence'],
            })

        # 오래된 프레임 정리 (MAX_BUFFER_SEC 초과분 제거)
        cutoff = timestamp - MAX_BUFFER_SEC
        while self.frames and self.frames[0]['t'] < cutoff:
            self.frames.popleft()

    def current_silence_duration(self, now=None):
        if self.silence_start_t is None:
            return 0.0
        # 주의: `now or time.time()`으로 쓰면 now=0.0(영상 시작 시점)이 falsy로 취급되어
        # 의도와 다르게 time.time()으로 대체되는 버그가 생김. 명시적으로 None 체크 필요.
        now = now if now is not None else time.time()
        return now - self.silence_start_t

    def is_significant_silence(self, now=None):
        return self.current_silence_duration(now) >= SILENCE_THRESHOLD_SEC

    def summarize(self):
        """
        버퍼에 쌓인 감정 프레임들을 요약.
        '2회 연속' 조건(Jarvis 로직)을 적용해 일시적 노이즈를 걸러내고,
        지배적 감정(dominant emotion)과 지속시간을 반환.
        """
        if not self.frames:
            return None

        # 트리거 감정만 추출해서 연속성 확인
        trigger_run = 0
        max_trigger_run = 0
        dominant_trigger = None
        emotion_counts = {}

        for f in self.frames:
            emotion_counts[f['emotion']] = emotion_counts.get(f['emotion'], 0) + 1

            if f['emotion'] in TRIGGER_EMOTIONS and f['confidence'] >= EMOTION_CONFIDENCE_THRESHOLD:
                trigger_run += 1
                if trigger_run > max_trigger_run:
                    max_trigger_run = trigger_run
                    dominant_trigger = f['emotion']
            else:
                trigger_run = 0

        duration = self.frames[-1]['t'] - self.frames[0]['t']

        return {
            'duration_sec': duration,
            'frame_count': len(self.frames),
            'emotion_distribution': emotion_counts,
            'should_trigger': max_trigger_run >= MIN_CONSECUTIVE_TRIGGERS,
            'dominant_trigger_emotion': dominant_trigger,
        }

    def flush(self):
        """발화가 재개되면 버퍼 초기화"""
        self.frames.clear()
        self.silence_start_t = None


# Jarvis 노트 기준 트리거 메시지 (그대로 재사용)
TRIGGER_MESSAGES = {
    '슬픔':   ["많이 힘드세요?", "무슨 일 있어요?", "표정이 좀 안 좋아 보여요. 괜찮으세요?"],
    '두려움': ["뭔가 걱정되는 게 있어요?", "불안해 보이는데 괜찮아요?"],
    '화남':   ["뭔가 속상한 일이 있었나요?", "화가 나 보이는데 무슨 일이에요?"],
    '혐오':   ["뭔가 불편한 게 있어요?", "표정이 좋지 않아 보여요."],
}


def build_silence_context_prompt(summary):
    """
    침묵 구간 요약을 EXAONE에 넘길 컨텍스트 문자열로 변환.
    MIPA의 핵심 novelty: 침묵 자체를 텍스트 컨텍스트로 명시적으로 전달.
    """
    if summary is None or not summary['should_trigger']:
        return None

    emotion = summary['dominant_trigger_emotion']
    duration = summary['duration_sec']

    context = (
        f"[비언어 신호] 사용자가 {duration:.1f}초간 말을 하지 않고 있으며, "
        f"이 침묵 구간 동안 '{emotion}' 감정이 우세하게 관찰되었습니다. "
        f"감정 분포: {summary['emotion_distribution']}."
    )
    return context


class FusionEngine:
    """
    perception 결과와 STT/VAD 상태를 받아 EXAONE에 넘길 컨텍스트를 관리하는 상위 레벨 클래스.
    실제 웹캠/마이크 스트림 루프에서 매 프레임 update()를 호출하는 형태로 사용 예정.
    """

    def __init__(self):
        self.silence_buffer = SilenceBuffer()
        self.is_user_speaking = False
        self._last_trigger_summary = None

    def on_speech_detected(self):
        """VAD가 발화 시작을 감지했을 때 호출"""
        self.is_user_speaking = True
        self.silence_buffer.flush()

    def on_speech_ended(self):
        """VAD가 발화 종료(침묵 시작)를 감지했을 때 호출"""
        self.is_user_speaking = False

    def update_perception(self, perception_result):
        """매 프레임 perception 결과를 받아 침묵 버퍼에 반영. 발화 중이면 버퍼링 안 함."""
        if self.is_user_speaking:
            return None

        self.silence_buffer.add_frame(perception_result)

        if self.silence_buffer.is_significant_silence():
            summary = self.silence_buffer.summarize()
            prompt_context = build_silence_context_prompt(summary)
            if prompt_context:
                self._last_trigger_summary = summary
                return prompt_context

        return None


if __name__ == '__main__':
    # 간단한 시뮬레이션 테스트: 실제 웹캠/마이크 없이 로직만 검증
    print("Fusion 레이어 로직 테스트 (시뮬레이션)\n")

    engine = FusionEngine()
    engine.on_speech_ended()  # 침묵 시작

    # 가짜 perception 결과를 3초에 걸쳐 흘려보내며 침묵 버퍼 테스트
    fake_results = [
        {'emotion': {'label_kr': '슬픔', 'confidence': 0.9}},
        {'emotion': {'label_kr': '슬픔', 'confidence': 0.85}},
        {'emotion': {'label_kr': '슬픔', 'confidence': 0.92}},
    ]

    for i, fr in enumerate(fake_results):
        # 침묵 시작 시점을 인위적으로 과거로 밀어서 SILENCE_THRESHOLD를 넘기도록 시뮬레이션
        engine.silence_buffer.silence_start_t = time.time() - 2.0
        result = engine.update_perception(fr)
        if result:
            print(f"[프레임 {i+1}] LLM 컨텍스트 트리거됨:")
            print(f"  {result}\n")
        else:
            print(f"[프레임 {i+1}] 트리거 조건 미충족")