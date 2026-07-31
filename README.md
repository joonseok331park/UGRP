# UGRP CAN IDS 연구 프로토타입

CAN 버스 로그를 토큰화하고 Transformer 교사 모델과 BiLSTM 학생 모델을 검토하기 위해 작성한 학부 연구 코드의 과거 스냅샷입니다. 구현이 완료된 제품이나 재현 패키지가 아니며, 현재 상태 그대로 실제 차량 또는 운영 환경에 배포할 수 없습니다.

## 저장소에 포함된 내용

| 경로 | 내용 |
| --- | --- |
| `core/tokenizer.py` | CAN ID와 페이로드 바이트를 위한 어휘 및 시퀀스 변환 |
| `core/dataset.py` | 로그 파싱과 분류용 시퀀스 데이터셋 초안 |
| `models/teacher.py` | BERT 기반 MLM 교사 모델 정의 |
| `models/student.py` | BiLSTM 학생 분류 모델 정의 |
| `scripts/prepare_dataset.py` | 여러 CAN 로그 파일을 하나의 파일로 합치는 데이터 준비 스크립트 |

교사 모델과 학생 모델을 연결하는 학습·지식 증류 루프는 구현되어 있지 않습니다. 원본 데이터, 생성된 어휘, 학습 체크포인트, 실행 로그와 평가 결과도 저장소에 포함하지 않습니다.

## 환경 설정

```bash
git clone https://github.com/joonseok331park/UGRP.git
cd UGRP
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows에서는 `.venv\Scripts\activate`로 가상 환경을 활성화합니다. 의존성 버전은 고정되어 있지 않으므로, 다시 실행할 때는 호환되는 PyTorch와 Transformers 조합을 별도로 확인해야 합니다.

## 의도한 데이터 흐름

`scripts/prepare_dataset.py`는 아래와 같은 상대 경로에서 `.log` 파일을 찾습니다.

```text
data/raw/can_mirgu/
├── Benign/
└── Attack/
```

데이터를 준비하는 명령은 다음과 같습니다.

```bash
python -m scripts.prepare_dataset
```

결과 파일은 `data/processed/`에 생성됩니다. 현재 저장소에는 이 결과를 받아 교사 모델을 종단 간 학습하는 진입점이 없습니다. 데이터 준비, 토크나이저와 모델 정의는 각각 별도의 프로토타입으로 남아 있습니다.

## 현재 확인된 제한

- 자동화된 종단 간 테스트나 지속적 통합 설정이 없습니다.
- 데이터 파싱, 토큰화, 학습 스크립트 사이의 인터페이스가 완전히 정리되지 않았습니다.
- 교사 모델 학습과 지식 증류를 실행하는 완성된 파이프라인이 없습니다.
- 정확도, F1, 모델 크기, 추론 시간에 관한 재현 가능한 결과를 제공하지 않습니다.
- 안전이 중요한 차량 시스템에 적용하려면 독립적인 데이터 검증, 위협 모델링, 오류 분석과 런타임 검증이 필요합니다.

이 저장소는 진행 중이던 연구 구현을 참고할 수 있도록 보존한 자료입니다. 결과가 검증된 완성 프로젝트로 해석해서는 안 됩니다.
