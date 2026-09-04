# vad-gen

영상의 발화 구간에는 대사를, 침묵 구간에는 표정과 제스처를 바탕으로 생성한 화면해설을 넣어 SRT 자막을 만드는 연구용 프로토타입입니다.

처리 순서는 다음과 같습니다.

```text
입력 영상
 ├─ POSTER++ 표정 인식 + MediaPipe 제스처 분석
 └─ faster-whisper 음성 인식 + Silero VAD 발화 구간 검출
             ↓
       시간축 기준 통합
             ↓
   EXAONE 3.5 화면해설 생성
             ↓
      SRT + 자막 영상
```

## 실행 환경

현재 구현을 확인한 환경은 아래와 같습니다.

- Ubuntu 22.04 LTS
- Python 3.10
- NVIDIA GeForce RTX 5060 Ti 16GB
- PyTorch 2.11.0 + CUDA 12.8
- ffmpeg

다른 NVIDIA GPU에서도 실행할 수 있지만, PyTorch 설치 명령은 GPU와 드라이버에 맞게 바꿔야 할 수 있습니다.

## 설치

### 1. 저장소와 시스템 패키지 준비

```bash
git clone https://github.com/bumkwanson/mipa-vad-gen-research.git
cd mipa-vad-gen-research

sudo apt update
sudo apt install -y python3.10-venv ffmpeg git
```

### 2. 가상환경 생성

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. PyTorch 설치

RTX 5060 Ti에서 확인한 CUDA 12.8 빌드는 다음과 같이 설치합니다.

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

설치 후 GPU 인식을 확인합니다.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

마지막 두 줄에 `True`와 GPU 이름이 표시되어야 합니다. 다른 환경용 설치 명령은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 확인할 수 있습니다.

### 4. 나머지 Python 패키지 설치

```bash
pip install -r requirements.txt
pip check
```

### 5. POSTER++ 가중치 다운로드

```bash
python scripts/download_weights.py
```

다운로드가 끝나면 다음 세 파일이 있어야 합니다.

```text
weights/
├── raf-db-model_best.pth
└── pretrain/
    ├── ir50.pth
    └── mobilefacenet_model_best.pth.tar
```

스크립트가 동작하지 않으면 [weights/README.md](weights/README.md)에 적힌 POSTER++ 공식 링크에서 직접 내려받아 같은 위치에 넣습니다.

### 6. Ollama와 EXAONE 준비

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull exaone3.5:7.8b
ollama list
```

실행 전에 Ollama 서버가 떠 있어야 합니다.

```bash
ollama serve
```

### 7. 설치 상태 확인

```bash
python scripts/check_setup.py
```

모든 필수 항목이 `OK`이면 실행 준비가 끝난 것입니다. Ollama 없이 결정론적 설명만 시험하려면 다음을 사용합니다.

```bash
python scripts/check_setup.py --skip-ollama
```

## 실행

10~30초 정도의 한 사람이 등장하는 영어 영상을 준비해 `samples/demo.mp4`로 둡니다. 영상은 저작권 문제로 저장소에 포함하지 않았습니다.

전체 파이프라인을 실행합니다.

```bash
python run.py samples/demo.mp4
```

첫 실행에서는 `faster-whisper`의 `large-v3-turbo` 모델을 내려받기 때문에 시간이 더 걸립니다.

Ollama 없이 고정된 표현만 사용하려면:

```bash
python run.py samples/demo.mp4 --deterministic
```

SRT만 만들고 자막 영상 생성을 건너뛰려면:

```bash
python run.py samples/demo.mp4 --srt-only
```

다른 가중치 폴더나 출력 폴더를 사용할 수도 있습니다.

```bash
python run.py samples/demo.mp4 \
  --weights-dir /path/to/weights \
  --output-dir /path/to/outputs
```

## 출력

기본 출력 위치는 `outputs/`입니다.

```text
outputs/
├── demo.srt
└── demo_subtitled.mp4
```

터미널에는 다음과 같이 구간별 결과가 표시됩니다.

```text
[  0.0s-  2.7s] SILENCE A startled expression.
[  2.7s-  4.7s] SPEECH  So can I ask you a question?
```

## 테스트

GPU 모델을 불러오지 않는 핵심 로직 테스트는 다음 명령으로 실행합니다.

```bash
python -m unittest discover -s tests -v
```

실제 전체 실행 확인에는 POSTER++ 가중치, Whisper 모델, 입력 영상이 필요하며 기본 모드에서는 Ollama도 필요합니다.

## 폴더 구조

```text
.
├── run.py                  전체 파이프라인 실행
├── requirements.txt
├── scripts/
│   ├── check_setup.py      설치 상태 확인
│   └── download_weights.py POSTER++ 가중치 다운로드
├── src/
│   ├── pipeline.py         영상 분석 흐름
│   ├── vision/             표정·제스처 분석
│   ├── audio/              STT·VAD
│   ├── align/              스무딩·시간축 통합
│   ├── generate/           화면해설 생성
│   └── output/             SRT·자막 영상 출력
├── experiments/            RAVDESS·Seamless 실험 스크립트
├── tests/                  핵심 로직 테스트
├── weights/                모델 가중치(별도 다운로드)
└── samples/                사용자 입력 영상
```

## 연구 범위

- 한 명이 등장하는 짧은 영어 영상의 파일 기반 배치 분석을 대상으로 합니다.
- 제스처 라벨은 MediaPipe 랜드마크에 적용한 규칙 기반 관찰값입니다.
- 침묵 구간의 정답은 사람 주석이 아니라 Silero VAD 결과입니다.
- `experiments/exp_silence_accuracy.py`의 비교 기준은 Seamless Interaction의 모델 출력이며 사람 주석 기반 정확도가 아닙니다.
- RAVDESS 실험은 데이터가 있는 경우에만 별도로 실행할 수 있습니다.

## 사용한 오픈소스

- [POSTER++](https://github.com/Talented-Q/POSTER_V2): 표정 인식
- [MediaPipe](https://github.com/google-ai-edge/mediapipe): 얼굴·자세·손 랜드마크
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): 음성 인식
- [Silero VAD](https://github.com/snakers4/silero-vad): 발화 구간 검출
- [EXAONE 3.5](https://ollama.com/library/exaone3.5): 화면해설 문장 생성

POSTER++에서 가져온 모델 구현의 라이선스는 [POSTER_LICENSE](POSTER_LICENSE)에 포함되어 있습니다.
