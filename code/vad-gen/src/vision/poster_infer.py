"""
POSTER++ 표정 인식 추론 래퍼 (vad-gen 프로젝트용)
얼굴 crop 이미지 -> 7-class 감정 확률
"""
import sys
import os
import collections.abc
import types
import builtins

import torch
if 'torch._six' not in sys.modules:
    _six = types.ModuleType('torch._six')
    _six.container_abcs = collections.abc
    _six.string_classes = (str, bytes)
    _six.int_classes = (int,)
    sys.modules['torch._six'] = _six

if not hasattr(builtins, 'RecorderMeter'):
    class RecorderMeter: pass
    class RecorderMeter1: pass
    builtins.RecorderMeter = RecorderMeter
    builtins.RecorderMeter1 = RecorderMeter1
    _m = sys.modules.get('__main__')
    if _m is not None:
        _m.RecorderMeter = RecorderMeter
        _m.RecorderMeter1 = RecorderMeter1

import cv2
import numpy as np
from torchvision import transforms

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from poster_model.PosterV2_7cls import pyramid_trans_expr2, load_pretrained_weights

LABELS = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

DEFAULT_CKPT = '/home/son/vad-gen/weights/raf-db-model_best.pth'


class PosterEmotion:
    def __init__(self, checkpoint_path=DEFAULT_CKPT, device='cuda'):
        self.device = device
        self.model = pyramid_trans_expr2(img_size=224, num_classes=7)
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model = load_pretrained_weights(self.model, ckpt)
        self.model = self.model.to(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, face_bgr):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        t = self.transform(rgb).unsqueeze(0).to(self.device)
        out = self.model(t)
        probs = torch.softmax(out.float(), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return {
            'label': LABELS[idx],
            'confidence': float(probs[idx]),
            'probs': {LABELS[i]: float(probs[i]) for i in range(7)}
        }


if __name__ == '__main__':
    m = PosterEmotion()
    print("POSTER++ 로드 성공")
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        print(m.predict(img))
