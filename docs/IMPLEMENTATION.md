# 실제 구현과 논문 표현

## vad-gen: 확인한 코드 경로

`test_export.py` → `test_timeline.analyze_video` → `AudioAnalyzer.analyze` → `build_timeline` → `DescriptionGenerator.generate` → `save_srt` → `burn_subtitles`.

| 요소 | 현재 설정 | 근거 |
|---|---|---|
| POSTER 표정 | Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral / FP32 | `src/vision/poster_infer.py` |
| 얼굴 전처리 | 검출 박스 여백 0.2, 224×224, ImageNet 정규화 | `test_timeline.py`, `poster_infer.py` |
| 시간 스무딩 | 영상 fps×0.4를 바탕으로 홀수 창, 최소 3프레임 | `test_timeline.py` |
| 짧은 구간 | 최소 0.8초, 앞 구간이 있으면 흡수 | `src/align/segment.py` |
| STT | large-v3-turbo, 영어, beam_size=5, 단어 시간, vad_filter=True | `src/audio/audio_analysis.py` |
| VAD | threshold=0.7, silence 최소 1.5초 | 동일 |
| 통합 표정 | 구간 겹침 길이를 누적한 최빈 감정 | `src/align/timeline.py` |
| 통합 제스처 | 구간 길이의 절반 이상 겹친 제스처 | 동일 |
| 단어 선택 | 발화 경계 앞뒤 0.15초 허용 | 동일 |
| LLM | Ollama exaone3.5:7.8b, temperature=0.1, 요청 timeout=120 | `src/generate/describe.py` |
| 출력 | 발화 텍스트, 변화한 침묵 상태 설명, SRT 및 자막 영상 | `describe.py`, `srt_export.py` |

## 논문 작성 전에 확인할 사항

1. `PLAN.md`의 작업 체크박스는 코드보다 오래되어 있습니다. 구현 유무는 실제 코드와 로그를 기준으로 씁니다.
2. 기존 SRT에는 1.5초보다 짧은 침묵 자막이 있습니다. 현재 AudioAnalyzer 기본값과 맞지 않으므로, 과거 산출물과 현재 코드가 같은 설정에서 생성되었다고 단정하지 않습니다.
3. `exp_silence_accuracy.py`는 `movement:emotion_scores` 모델 출력을 비교 기준으로 삼습니다. 사람 주석 기반 정확도가 아니라 모델 간 일치율로 구분해야 합니다. 주석 모델·버전과 프레임 정렬 검증이 필요합니다.
4. `exp_ravdess.py`는 Actor_01의 `01-01-*`만 사용하고 Calm을 제외합니다. 얼굴 검출 성공 후 예측이 있는 클립만 분모에 들어갑니다. 전체 데이터셋 일반 성능으로 보고할 수 없습니다.
5. 얼굴 미검출 프레임을 제외하면 결과가 낙관적일 수 있습니다. 전체 대상 수, 검출 실패 수, 평가에 포함된 수를 함께 보고해야 합니다.
6. 스무딩이 미래 프레임도 보는 중앙 창이므로 vad-gen은 파일 기반 배치 분석으로 설명합니다.
7. 금칙어 검사는 문자열 부분 일치입니다. 정상 관찰 어구에도 포함되는 `face`, `lowered` 등이 차단될 수 있습니다. LLM 출력과 대체 문장 비율을 기록해야 하며 환각이 완전히 제거되었다고 주장하지 않습니다.
8. 모델 프롬프트에 없는 인물 속성·내적 감정을 사실로 서술하지 않습니다. 기하 규칙은 자세에 대한 추정이며 심리 상태 검증이 아닙니다.

## MIPA 초기 구현과의 차이

- `realtime_webcam.py`는 웹캠 1번을 고정 사용하고 실제 오디오 입력/VAD를 연결하지 않습니다. 초기부터 침묵 상태로 두는 데모입니다.
- `fusion.py`는 감정 라벨과 confidence를 저장하지만 pose는 버퍼에 저장하지 않습니다.
- 부정 감정의 2회 연속 조건은 동일 감정의 연속일 필요가 없습니다. `dominant_trigger_emotion`은 단순 최빈 감정이 아닙니다.
- 추출 감정 프레임의 시간 폭과 전체 침묵 시간은 다를 수 있습니다.
- 초기 코드의 `inference.py`는 현재 FP16 기본값이 켜져 있습니다. 공유 대화의 “롤백” 설명만으로 FP32라고 기록하지 않습니다.
- 실시간 LLM 호출은 동기 방식이며 호출 중 영상 루프가 대기합니다. TTS·barge-in 완성 시스템으로 표현하지 않습니다.
