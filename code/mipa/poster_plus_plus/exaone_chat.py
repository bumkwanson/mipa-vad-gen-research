"""
EXAONE 3.5 7.8B 연동 모듈 (Ollama 기반)
transformers 직접 로드 시 발생하는 버전 호환성/VRAM 피크 문제를 피하기 위해
Ollama의 HTTP API를 사용. Jarvis 프로젝트에서 검증된 방식과 동일.

사전 준비:
    curl -fsSL https://ollama.ai/install.sh | sh
    ollama pull exaone3.5:7.8b
    ollama serve &   (보통 설치 시 자동으로 백그라운드 서비스 등록됨)
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "exaone3.5:7.8b"

SYSTEM_PROMPT = (
    "당신은 사용자의 표정과 침묵을 관찰하며 대화하는 다정한 AI 동반자입니다. "
    "사용자가 아무 말도 하지 않고 있을 때, 관찰된 감정 신호를 바탕으로 "
    "짧고 자연스럽게 먼저 말을 건넵니다. 과하게 걱정하거나 부담스럽지 않게, "
    "한두 문장으로 담백하게 물어보세요."
)


class ExaoneChat:
    def __init__(self, model_name=MODEL_NAME, base_url=OLLAMA_URL):
        self.model_name = model_name
        self.base_url = base_url
        self.history = []

        self._check_server()

    def _check_server(self):
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model_name not in models:
                print(f"경고: '{self.model_name}' 모델이 Ollama에 없습니다. "
                      f"'ollama pull {self.model_name}'로 먼저 받아주세요.")
                print(f"현재 받아진 모델 목록: {models}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Ollama 서버에 연결할 수 없습니다. 'ollama serve'가 실행 중인지 확인하세요."
            )

    def generate(self, content, add_to_history=True):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": content})

        response = requests.post(self.base_url, json={
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        })
        response.raise_for_status()
        result = response.json()
        reply = result["message"]["content"].strip()

        if add_to_history:
            self.history.append({"role": "user", "content": content})
            self.history.append({"role": "assistant", "content": reply})
            if len(self.history) > 12:
                self.history = self.history[-12:]

        return reply

    def reset_history(self):
        self.history = []


if __name__ == '__main__':
    import time

    print("Ollama 연결 확인 중...")
    chat = ExaoneChat()
    print("연결 완료\n")

    test_context = (
        "[비언어 신호] 사용자가 4.8초간 말을 하지 않고 있으며, "
        "이 침묵 구간 동안 '슬픔' 감정이 우세하게 관찰되었습니다. "
        "감정 분포: {'평온': 32, '기쁨': 21, '혐오': 1, '슬픔': 1}."
    )

    print(f"입력 컨텍스트: {test_context}\n")

    t0 = time.time()
    response = chat.generate(test_context)
    gen_time = time.time() - t0

    print(f"응답 ({gen_time:.1f}초 소요): {response}")