# 논문 작성 자료 묶음

2026-09-04 수집. 논문 작업 초기 구현과 vad-gen 후속 구현의 코드·기존 산출물·연구 계획
## 먼저 읽을 문서

| 자료 | 내용 |
|---|---|
| [논문 자료 안내](docs/PAPER_MATERIALS.md) | 논문 각 장에 사용할 자료와 부족한 근거 |
| [구현 상태와 한계](docs/IMPLEMENTATION.md) | 실제 코드로 확인한 기능과 주의할 해석 |
| [실험 계획 및 결과 관리](docs/EXPERIMENTS.md) | 기존 실험의 범위, 추가 평가, 결과 표 양식 |
| [설치·실행 현황](docs/REPRODUCIBILITY.md) | 실행 진입점, 의존성 충돌, 재현을 막는 항목 |
| [자료 보관·배포](docs/ASSETS.md) | 모델·데이터 원본 위치와 외부 전달 준비 |
| [참고문헌 출발 목록](references/README.md) | 확인한 논문·공식 구현 링크 |
| [전체 파일 목록](manifests/files.csv) | 원본 위치, 크기, 복사 위치, 일부 SHA-256 |

## 폴더 구조

```text
code/mipa/              초기 연구 코드와 원본 POSTER 문서
code/vad-gen/           후속 연구 코드, 실험 스크립트, PLAN.md, 과거 백업
local_materials/       기존 SRT·데모·로그·이미지, 연구일지, 투고 양식
environment/           현재 설치된 패키지와 진단 결과
evidence/              원본 Git 변경 내역, 출처, 이번 점검 기록
manifests/             대용량 원본까지 포함한 파일별 목록
references/            논문·모델·데이터 공식 출처
tools/                 가벼운 재현 확인과 원본 자료 복사 도구
```

코드 사본은 원본 그대로 보존했습니다. `local_materials`는 개인 연구용 자료이며 Git 기본 제외 대상으로 설정했습니다. 대용량 가중치·데이터는 중복 복사하지 않고 원본을 목록에 등록했습니다. 이 묶음만 다른 컴퓨터로 옮겨도 GPU 실험 전체가 실행되는 상태는 아닙니다.


```bash
python3 tools/smoke_test.py
```

성공하면 `PASS`와 합성 SRT가 출력 (표정·음성 모델이나 LLM 품질 검증 X)

이번 점검에서 Python 파일 53개의 문법 확인, 복사 파일 95개의 원본 대비 SHA-256 일치, 합성 타임라인→SRT 확인을 통과했습니다. 상세 결과는 `evidence/verification.json`과 `evidence/smoke_test.txt`에 있습니다.
