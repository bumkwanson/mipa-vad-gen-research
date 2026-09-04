"""
POSTER++ 표정 인식 추론 래퍼 (vad-gen 프로젝트용)
얼굴 crop 이미지 -> 7-class 감정 확률
"""

import builtins
import collections.abc
import os
import sys
import types
from pathlib import Path

import torch

if "torch._six" not in sys.modules:
    _six = types.ModuleType("torch._six")
    _six.container_abcs = collections.abc
    _six.string_classes = (str, bytes)
    _six.int_classes = (int,)
    sys.modules["torch._six"] = _six

if not hasattr(builtins, "RecorderMeter"):

    class RecorderMeter:
        pass

    class RecorderMeter1:
        pass

    builtins.RecorderMeter = RecorderMeter
    builtins.RecorderMeter1 = RecorderMeter1
    _m = sys.modules.get("__main__")
    if _m is not None:
        _m.RecorderMeter = RecorderMeter
        _m.RecorderMeter1 = RecorderMeter1

import cv2
import numpy as np
from torchvision import transforms

from .poster_model.PosterV2_7cls import load_pretrained_weights, pyramid_trans_expr2

LABELS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"


def _resolve_weights_dir(weights_dir):
    configured = (
        weights_dir or os.environ.get("VADGEN_WEIGHTS_DIR") or DEFAULT_WEIGHTS_DIR
    )
    return Path(configured).expanduser().resolve()


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 장치를 요청했지만 PyTorch가 GPU를 인식하지 못합니다.")
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"지원하지 않는 장치입니다: {device}")
    return device


class PosterEmotion:
    def __init__(self, checkpoint_path=None, weights_dir=None, device="auto"):
        weights_root = _resolve_weights_dir(weights_dir)
        checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path
            else weights_root / "raf-db-model_best.pth"
        )
        mobileface_path = weights_root / "pretrain" / "mobilefacenet_model_best.pth.tar"
        ir50_path = weights_root / "pretrain" / "ir50.pth"

        missing = [
            path
            for path in (checkpoint_path, mobileface_path, ir50_path)
            if not path.is_file()
        ]
        if missing:
            lines = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "POSTER++ 가중치가 없습니다. `python scripts/download_weights.py`를 실행하거나 "
                "--weights-dir를 지정하세요.\n" + lines
            )

        self.device = _resolve_device(device)
        self.model = pyramid_trans_expr2(
            img_size=224,
            num_classes=7,
            mobileface_checkpoint_path=mobileface_path,
            ir_checkpoint_path=ir50_path,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = load_pretrained_weights(self.model, ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def predict(self, face_bgr):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        t = self.transform(rgb).unsqueeze(0).to(self.device)
        out = self.model(t)
        probs = torch.softmax(out.float(), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return {
            "label": LABELS[idx],
            "confidence": float(probs[idx]),
            "probs": {LABELS[i]: float(probs[i]) for i in range(7)},
        }


if __name__ == "__main__":
    m = PosterEmotion()
    print("POSTER++ 로드 성공")
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        print(m.predict(img))
