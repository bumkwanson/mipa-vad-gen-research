"""
POSTER++ 실시간 추론 래퍼
웹캠 프레임(numpy array, BGR) -> 감정 확률 벡터(7 classes)

RAF-DB 7-class 라벨 순서 (POSTER_V2 공식 기준):
0: Surprise, 1: Fear, 2: Disgust, 3: Happiness, 4: Sadness, 5: Anger, 6: Neutral
"""
import sys
import collections.abc
import torch

# timm==0.3.2는 PyTorch 1.x 시절 torch._six.container_abcs를 참조하는데,
# 최신 PyTorch(1.9+)에서 이 모듈이 완전히 제거됨. 가짜 모듈을 등록해서 우회.
if 'torch._six' not in sys.modules:
    import types
    _six_shim = types.ModuleType('torch._six')
    _six_shim.container_abcs = collections.abc
    _six_shim.string_classes = (str, bytes)
    _six_shim.int_classes = (int,)
    sys.modules['torch._six'] = _six_shim

import builtins

# Jarvis 프로젝트 노트에서 확인된 알려진 함정:
# 체크포인트가 pickle될 때 RecorderMeter/RecorderMeter1이 __main__ 모듈 소속으로 저장되어 있어서,
# 이 파일이 다른 스크립트(perception.py 등)에서 import될 때는 __main__이 그 스크립트가 되어
# builtins 등록만으로는 unpickler가 못 찾음. sys.modules['__main__']에도 직접 등록.
if not hasattr(builtins, 'RecorderMeter'):
    class RecorderMeter:
        pass

    class RecorderMeter1:
        pass

    builtins.RecorderMeter = RecorderMeter
    builtins.RecorderMeter1 = RecorderMeter1

    _main_module = sys.modules.get('__main__')
    if _main_module is not None:
        if not hasattr(_main_module, 'RecorderMeter'):
            _main_module.RecorderMeter = RecorderMeter
        if not hasattr(_main_module, 'RecorderMeter1'):
            _main_module.RecorderMeter1 = RecorderMeter1

import torch
import cv2
import numpy as np
from torchvision import transforms
from models.PosterV2_7cls import pyramid_trans_expr2, load_pretrained_weights

LABELS = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
LABELS_KR = ['놀람', '두려움', '혐오', '기쁨', '슬픔', '화남', '평온']


class PosterEmotionRecognizer:
    def __init__(self, checkpoint_path, device='cuda', use_fp16=True):
        self.device = device
        # RTX 5060 Ti(Blackwell)는 FP16 연산이 FP32보다 훨씬 빠름.
        # CPU 추론일 땐 FP16 지원이 불안정할 수 있어 GPU일 때만 적용.
        self.use_fp16 = use_fp16 and device == 'cuda'

        self.model = pyramid_trans_expr2(img_size=224, num_classes=7)

        # PyTorch 2.6+ 부터 torch.load 기본값이 weights_only=True로 바뀌어서
        # POSTER_V2 체크포인트(임의 객체 포함)를 그대로 못 읽음. 명시적으로 False 지정.
        # (신뢰할 수 있는 출처의 체크포인트일 때만 안전 — 우리가 직접 받은 공식 weight라 문제없음)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model = load_pretrained_weights(self.model, checkpoint)

        self.model = self.model.to(device)
        if self.use_fp16:
            self.model = self.model.half()
        self.model.eval()

        # POSTER_V2 학습 시 사용한 정규화 값 (ImageNet 기준)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, frame_bgr):
        """
        frame_bgr: 이미 얼굴 crop된 BGR numpy 이미지 (H, W, 3)
        반환: {'label': str, 'confidence': float, 'probs': dict}
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        if self.use_fp16:
            tensor = tensor.half()

        output = self.model(tensor)
        # softmax는 FP16보다 FP32가 안정적이라 여기서 다시 float()로 변환
        probs = torch.softmax(output.float(), dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        return {
            'label': LABELS[pred_idx],
            'label_kr': LABELS_KR[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probs': {LABELS[i]: float(probs[i]) for i in range(7)}
        }


if __name__ == '__main__':
    import sys

    ckpt = './checkpoint/raf-db-model_best.pth'
    recognizer = PosterEmotionRecognizer(ckpt, device='cuda' if torch.cuda.is_available() else 'cpu')
    print("모델 로드 완료")

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지를 읽을 수 없음: {img_path}")
            sys.exit(1)
        result = recognizer.predict(img)
        print(result)
    else:
        print("사용법: python inference.py <이미지경로>")
        print("이미지 없이 모델 로드 테스트만 완료.")