"""전체 추론을 시작하기 전에 설치 상태를 빠르게 확인합니다."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_WEIGHTS = (
    Path("raf-db-model_best.pth"),
    Path("pretrain/ir50.pth"),
    Path("pretrain/mobilefacenet_model_best.pth.tar"),
)


def report(status: str, label: str, detail: str) -> None:
    print(f"[{status:<4}] {label}: {detail}")


def import_version(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - 진단 스크립트는 import 중 발생한 예외도 보고해야 한다.
        return False, str(exc)
    return True, str(getattr(module, "__version__", "installed"))


def ollama_tags_url(chat_url: str) -> str:
    base = chat_url.split("/api/", 1)[0].rstrip("/")
    return f"{base}/api/tags"


def main() -> int:
    parser = argparse.ArgumentParser(description="vad-gen 설치 상태를 확인합니다.")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=PROJECT_ROOT / "weights",
        help="POSTER++ 가중치 폴더",
    )
    parser.add_argument("--ollama-model", default="exaone3.5:7.8b")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/chat")
    parser.add_argument("--skip-ollama", action="store_true")
    args = parser.parse_args()

    failures = 0
    version = sys.version_info
    if version[:2] == (3, 10):
        report("OK", "Python", sys.version.split()[0])
    else:
        failures += 1
        report("FAIL", "Python", f"{sys.version.split()[0]} (3.10 필요)")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        report("OK", "ffmpeg", ffmpeg)
    else:
        failures += 1
        report("FAIL", "ffmpeg", "설치되지 않음")

    weights_dir = args.weights_dir.expanduser().resolve()
    missing = [
        str(path) for path in REQUIRED_WEIGHTS if not (weights_dir / path).is_file()
    ]
    if missing:
        failures += 1
        report("FAIL", "POSTER++ 가중치", ", ".join(missing) + " 없음")
    else:
        report("OK", "POSTER++ 가중치", str(weights_dir))

    modules = (
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("faster_whisper", "faster-whisper"),
        ("silero_vad", "Silero VAD"),
        ("soundfile", "soundfile"),
        ("requests", "requests"),
        ("gdown", "gdown"),
        ("src.vision.poster_infer", "POSTER++ 모듈"),
    )
    imported: dict[str, object] = {}
    for module_name, label in modules:
        ok, detail = import_version(module_name)
        if ok:
            report("OK", label, detail)
            imported[module_name] = sys.modules[module_name]
        else:
            failures += 1
            report("FAIL", label, detail)

    torch_module = imported.get("torch")
    if torch_module is not None:
        cuda_available = bool(torch_module.cuda.is_available())
        if cuda_available:
            report("OK", "CUDA", torch_module.cuda.get_device_name(0))
        else:
            report("WARN", "CUDA", "GPU를 인식하지 못함; CPU 실행은 매우 느릴 수 있음")

    if args.skip_ollama:
        report("SKIP", "Ollama", "--skip-ollama")
    else:
        try:
            with urlopen(ollama_tags_url(args.ollama_url), timeout=3) as response:
                payload = json.load(response)
            installed = {item.get("name", "") for item in payload.get("models", [])}
            if args.ollama_model in installed:
                report("OK", "Ollama", args.ollama_model)
            else:
                failures += 1
                report("FAIL", "Ollama", f"{args.ollama_model} 없음; ollama pull 필요")
        except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
            failures += 1
            report("FAIL", "Ollama", f"서버에 연결할 수 없음 ({exc})")

    if failures:
        print(f"\n필수 확인 {failures}개가 실패했습니다.")
        return 1

    print("\n실행 준비가 완료되었습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
