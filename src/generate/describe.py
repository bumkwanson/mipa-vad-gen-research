"""
화면해설(Audio Description) 생성 모듈 (vad-gen) - 변화 기반

핵심 원칙 (실제 AD 관례):
- 비언어 상태(표정+제스처)가 '바뀔 때만' 설명을 넣는다.
- 상태가 그대로 유지되면 설명을 생략한다 (침묵 유지).
- SPEECH 구간은 대사 그대로.
- 엄격 모드: 제공된 신호에 없는 디테일은 지어내지 않음.
"""

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "exaone3.5:7.8b"

EMOTION_HINT = {
    "Anger": "a tense, hardened expression",
    "Disgust": "a strained expression",
    "Fear": "an unsettled expression",
    "Happiness": "a light expression",
    "Sadness": "a downcast expression",
    "Surprise": "a startled expression",
    "Neutral": "a neutral expression",
}

GESTURE_HINT = {
    "hand_near_face": "a hand raised near the face",
    "hand_near_neck": "a hand at the neck",
    "head_turned": "the head turned away",
    "head_down": "the head lowered",
    "hands_lowered": "hands down at the sides",
}

SYSTEM_PROMPT = """You are a professional audio describer for blind and low-vision viewers.

You are given a FIXED LIST of CURRENT visual observations.

Write ONE short sentence describing only what is now visible.

ABSOLUTE RULES:
1. Use ONLY the current observations provided. NEVER add body parts, movements, or details
   not explicitly listed (no brows, eyes, lips, jaw, shoulders, breathing, etc.).
2. Never assert emotions or inner states. Keep expression wording as given ("tense", "downcast").
   Do not describe mood, atmosphere, intention, or cause.
3. Present tense, third person. Roughly 6-12 words.
4. Describe the current state without implying movement, change, or a previous state.
5. Output ONLY the sentence. No quotes, no labels."""


# 프롬프트만으로 막기 어려운 환각/변화 추론 표현. 발견 시 결정론적 안전 문장으로 대체.
FORBIDDEN_OUTPUT_TERMS = (
    "previous",
    "before",
    "from the",
    "move",
    "moving",
    "atmosphere",
    "mood",
    "feel",
    "intent",
    "because",
    "brow",
    "eye",
    "lip",
    "jaw",
    "shoulder",
    "breath",
    "collar",
)


def _observation_list(item):
    obs = []
    if item.get("emotion"):
        obs.append(EMOTION_HINT.get(item["emotion"], item["emotion"].lower()))
    for g in sorted(item.get("gestures", [])):
        obs.append(GESTURE_HINT.get(g, g.replace("_", " ")))
    return obs


def _state_key(item):
    """비언어 상태를 비교용 키로 (표정 + 정렬된 제스처)"""
    return (item.get("emotion"), tuple(sorted(item.get("gestures", []))))


def _safe_description(item):
    """감지된 고정 어구만 조합하는 엄격 모드 fallback."""
    obs = _observation_list(item)
    if not obs:
        return "No distinct visual signal is detected."
    sentence = "; ".join(obs)
    return sentence[0].upper() + sentence[1:] + "."


def _passes_guardrail(text):
    lowered = text.lower()
    return not any(term in lowered for term in FORBIDDEN_OUTPUT_TERMS)


class DescriptionGenerator:
    def __init__(
        self, model=DEFAULT_MODEL, base_url=OLLAMA_URL, use_llm=True, timeout=120
    ):
        self.model = model
        self.base_url = base_url
        self.use_llm = use_llm
        self.timeout = timeout

    def ensure_available(self):
        """LLM 모드일 때 Ollama 서버와 지정 모델을 실행 전에 확인합니다."""
        if not self.use_llm:
            return

        import requests

        tags_url = self.base_url.split("/api/", 1)[0].rstrip("/") + "/api/tags"
        try:
            response = requests.get(tags_url, timeout=3)
            response.raise_for_status()
            names = {
                name
                for item in response.json().get("models", [])
                for name in (item.get("name"), item.get("model"))
                if name
            }
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                "Ollama 서버에 연결할 수 없습니다. `ollama serve`를 실행하거나 "
                "--deterministic 옵션을 사용하세요."
            ) from exc

        if self.model not in names:
            raise RuntimeError(
                f"Ollama 모델 `{self.model}`이 없습니다. `ollama pull {self.model}`을 실행하세요."
            )

    def _gen_one(self, item, temperature=0.1):
        if not self.use_llm:
            return _safe_description(item)

        import requests

        cur_obs = _observation_list(item)
        cur_text = "; ".join(cur_obs) if cur_obs else "no distinct signals"
        user = (
            f"Current observations (use ONLY these): {cur_text}.\n\n"
            "Describe the current visible state without implying movement or a previous state."
        )
        try:
            resp = requests.post(
                self.base_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            desc = resp.json()["message"]["content"].strip().strip('"').strip()
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Ollama 화면해설 생성에 실패했습니다: {exc}") from exc
        return desc if _passes_guardrail(desc) else _safe_description(item)

    def generate(self, timeline):
        out = []
        last_described_state = None

        for item in timeline:
            new_item = dict(item)

            if item["type"] == "speech":
                new_item["description"] = item["text"]
            else:
                cur_state = _state_key(item)
                if cur_state != last_described_state:
                    # 상태가 바뀐 침묵 구간에만 설명 생성
                    desc = self._gen_one(item)
                    new_item["description"] = desc
                    last_described_state = cur_state
                else:
                    # 상태 그대로 -> 설명 생략
                    new_item["description"] = ""

            out.append(new_item)
        return out


def format_described(timeline):
    lines = []
    for item in timeline:
        tag = "SPEECH " if item["type"] == "speech" else "SILENCE"
        desc = item.get("description", "")
        marker = (
            desc
            if desc
            else "(no description - unchanged)"
            if item["type"] == "silence"
            else ""
        )
        lines.append(f"[{item['start']:5.1f}s-{item['end']:5.1f}s] {tag} {marker}")
    return "\n".join(lines)
