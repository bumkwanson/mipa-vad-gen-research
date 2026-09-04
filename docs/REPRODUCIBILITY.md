# 설치·재현 현황

이 문서는 현재 실행 진입점과 재현 장애를 기록합니다. 새 컴퓨터 설치를 보증하는 최종 README가 아닙니다.

## 이번 확인

- 기존 가상환경 Python: 3.10.20.
- 관측 실행 환경 OS: Ubuntu 24.04.4. 과거 프로젝트의 Ubuntu 22.04 기록과 구분해야 합니다.
- 패키지 목록: `environment/installed_packages.json`.
- `pip check` 실패: MediaPipe 0.10.21은 numpy<2를 요구하지만 NumPy 2.2.6이 설치되어 있습니다.
- OpenCV 기본·contrib 배포판이 함께 설치되어 있습니다. 새 환경에서는 동일 `cv2`를 공유하는 구성을 정리하고 검증해야 합니다.
- `nvidia-smi` 실패: 이번 에이전트 실행 환경에서 NVIDIA 드라이버와 통신되지 않습니다. 실제 호스트 GPU 고장으로 단정할 수 없습니다.
- 원본 POSTER requirements는 연구자가 추가한 전체 시스템의 설치 명세가 아닙니다. 관측 패키지 목록도 그대로 설치할 권장 lock 파일이 아닙니다.

## 실행 경로 (기존 원본 환경 기준)

```bash
cd /home/son/vad-gen
source /home/son/mipa/poster_plus_plus/venv_poster/bin/activate
python test_export.py data/test_clip.mp4
```

필요 요소: NVIDIA CUDA 실행 환경, POSTER 가중치 3개, MediaPipe, faster-whisper, Silero VAD, soundfile, requests, timm, thop, PyTorch·torchvision, ffmpeg, 실행 중인 로컬 Ollama와 `exaone3.5:7.8b`. Whisper/Silero 최초 실행에는 모델 다운로드가 발생할 수 있습니다.

원본 실행은 같은 이름의 `outputs/test_clip.srt`, `outputs/test_clip_subtitled.mp4`를 덮어씁니다. 연구 결과를 보존한 뒤 별도 실행 폴더를 사용해야 합니다.

표정 실험 진입점:

```bash
python exp_ravdess.py
python exp_silence_accuracy.py
```

각 스크립트는 내부의 대상 경로·목록을 고정 사용합니다. 데이터가 없을 때 안정적으로 종료하도록 구현되어 있지 않은 부분도 있어 실행 전 대상 목록 점검이 필요합니다.

## 외부 컴퓨터 재현을 막는 항목

1. `src/vision/poster_infer.py`의 체크포인트 기본값 및 `src/vision/poster_model/PosterV2_7cls.py`의 사전학습 모델 경로가 `/home/son/vad-gen/weights/...`로 고정되어 있습니다. 설정 또는 저장소 상대경로로 변경 필요.
2. 모델 3개 다운로드·배치 방법, 데이터 출처·클립 추출 시점·checksum을 고정한 안내가 필요합니다.
3. 원본 virtualenv 심볼릭 링크는 다른 컴퓨터에서 사용할 수 없습니다. 전용 환경과 검증된 의존성 명세 필요.
4. Silero torch.hub 호출이 master를 사용합니다. 실험 버전 고정 필요.
5. 코드·데이터·LLM 설정과 과거 출력의 연결이 완전하지 않습니다. 신규 실행별 메타데이터 저장 필요.

## 교수님 제출 완료 기준

새 폴더/환경에서 설치 → 가중치·샘플 준비 → 전체 실행 → SRT·영상 생성 → 결과 확인을 README만으로 수행하고 실행 로그를 보관해야 합니다. 이번 작업은 자료 수집 단계로, 위 GPU 전체 실행은 아직 수행하지 않았습니다.
