# 소프트웨어 설계 및 실행 계획서: UGRP

**[변경 로그 v1.1]**
- **변경 일자**: 2025-08-04
- **변경 사항**:
    - 사용자와의 논의를 통해 **목표 ECU 환경, 실시간 처리 기준, 학생 모델 아키텍처, 지식 증류 하이퍼파라미터** 등 핵심 설계 요소를 구체화하고 문서에 명시함.
    - 선제적 리스크 관리를 위해 **`4. 아키텍처 리스크 분석 및 개선 제안`** 섹션을 공식적으로 추가하고 대응 방안을 구체화함.
    - 데이터베이스 및 API 섹션을 본 프로젝트에 적합한 **`5. 데이터셋 및 산출물 관리`** 와 **`6. 실험 프로토콜 및 평가 지표`** 로 변경함.
    - 프로젝트 실행을 위한 **`7. 디렉토리 구조`** 부터 **`12. 설치 및 실행 방법`** 까지 모든 섹션을 작성하여 완전한 실행 계획서의 초안을 완성함.
- **변경자**: Specification (AI 수석 아키텍트)

## 1. 프로젝트 개요

- **1.1. 프로젝트 목적**: `CAN-BERT`의 지식을 경량 '학생 모델'에 증류하여, 실제 차량 ECU 환경에서 실시간으로 동작 가능한 고성능-고효율 차량 침입 탐지 시스템(IDS)을 완성한다.
- **1.2. 핵심 성공 지표 (Key Success Factors)**:
    - **성능**: Fuzzy 공격 F1-Score 기준, 교사 모델 대비 성능 저하를 **5% 이내**로 최소화한다. **(가장 중요한 전제 조건)**
    - **효율 (속도)**: 단일 시퀀스(Window=32) 추론 시간을 **0.1ms 이내**로 달성 (교사 모델 대비 10배 이상 향상).
    - **효율 (크기)**: 최종 모델의 저장 공간 점유율을 **16MB 미만**으로 경량화 (교사 모델 대비 1/5 이하).
- **1.3. 대상 사용자**: 자율주행 차량의 보안 시스템 및 보안 관제 연구원
- **1.4. 목표 시스템 제약 조건 (Target System Constraints)**:
    - **프로세서**: ARM Cortex-A 계열 (e.g., A53/A55)
    - **메모리 (RAM)**: 최대 사용 가능 메모리 **128MB** 제한
    - **저장 공간 (Storage)**: 최대 사용 가능 공간 **16MB** 제한

## 2. 기술 스택 (Technology Stack)

- **언어**: Python 3.12
- **핵심 프레임워크**: PyTorch
- **교사 모델**: `CAN-BERT` (BERT-base 아키텍처 기반)
- **학생 모델**: `BiLSTM` (상세 아키텍처는 섹션 3.4 참조)
- **주요 라이브러리**: Transformers, Scikit-learn, Pandas, NumPy, wandb
- **실험 관리 도구**: (제안) MLflow 또는 Weights & Biases (W&B) - 하이퍼파라미터 튜닝 및 결과 추적

## 3. 시스템 아키텍처

- **3.1. 아키텍처 패턴**: 교사-학생(Teacher-Student) 모델 기반의 지식 증류(Knowledge Distillation)
- **3.2. 고수준 설계도**:

    ## Step 1: 데이터 전처리 (Data Preprocessing)
    - 다양한 원본 CAN 데이터셋(CAN-MIRGU, HCRL 등)을 입력받습니다.
    - Jo & Kim (2024) 논문에 기반한 **CANTokenizer**와 **CANSequencer**를 사용하여, 모델이 학습할 수 있는 통합 시퀀스 데이터로 가공합니다.
    - 이 단계에서 **'오프셋 기반 통합 어휘집'**이 생성됩니다.

    ## Step 2: 교사 모델 사전 훈련 (Teacher Pre-training)
    - 가공된 대규모 정상 주행 데이터(CAN-MIRGU)를 사용하여, **CAN-BERT 교사 모델**의 사전 훈련을 진행합니다.
    - 이 과정을 통해 교사 모델은 CAN 통신의 일반적인 **'문법'과 '문맥'**을 학습하게 됩니다.
    - **참고 논문**: CAN-BERT do it?

    ## Step 3: 교사 모델 미세 조정 (Teacher Fine-tuning)
    - 사전 훈련된 교사 모델을, 레이블이 있는 공격 데이터(HCRL 등)로 미세 조정합니다.
    - 이 과정을 통해 교사 모델은 정상과 공격을 구분하는 **'전문 지식'**을 갖추게 됩니다.
    - 최종적으로 **성능은 높지만 무거운 교사 모델**이 완성됩니다.

    ## Step 4: 학생 모델 지식 증류 (Student Knowledge Distillation)
    > **이 단계가 우리 프로젝트의 핵심입니다.**
    - **가볍고 빠른 학생 모델(BiLSTM)**을 준비합니다.
    - 완성된 교사 모델의 예측(**Soft Label**)과 실제 정답(**Hard Label**)을 모두 활용하여 학생 모델을 훈련시킵니다.
    - 학생 모델은 교사의 **'판단 노하우'**를 전수받아, 작은 크기에도 불구하고 높은 성능을 갖추게 됩니다.
    - **참고 논문**: LSF-IDM

    ## Step 5: 최종 평가 (Final Evaluation)
    - 지식 증류가 완료된 경량 학생 모델을, 훈련에 사용되지 않은 완전히 새로운 데이터셋(can-train-and-test 등)으로 평가합니다.
    - 이 최종 평가를 통해, 우리는 **명세서 1.2. 핵심 성공 지표**에서 정의한 성능(F1-Score), 속도(Latency), 크기(Size) 목표를 달성했는지 최종 검증합니다.

- **3.3. 구성요소별 설명**:
    - **데이터 전처리**: '오프셋 기반 통합 어휘집' 및 'ID+Payload 통합 시퀀싱' 적용.
    - **교사 모델**: CAN-BERT. MLM 태스크로 사전 훈련하여 CAN 시퀀스의 문맥적 이해 능력 확보.
    - **학생 모델**: BiLSTM. 교사 모델의 지식을 증류 받아 경량 환경에서 효율적으로 동작.

- **3.4. 학생 모델 상세 설계 (Student Model Detailed Design)**:
    - **기반**: Bi-directional LSTM
    - **레이어 수**: 2-Layer Stacked BiLSTM
    - **Hidden State 차원**: 64
    - **Dropout 비율**: 0.2 (BiLSTM 출력 후, 최종 분류 레이어 전)

- **3.5. 지식 증류 학습 전략 (Knowledge Distillation Strategy)**:
    - **손실 함수**: L = α * L_CE + (1-α) * L_KD (L_CE (분류 손실): 학생이 실제 정답(Hard Label)을 맞추도록 유도, L_KD (지식 증류 손실): 학생이 교사의 Soft Label을 모방하도록 유도)
    - **α (Alpha) 초기값**: 0.5, [0.3, 0.7] 범위에서 튜닝 예정.
    - **T (Temperature) 초기값**: 2, [2, 5] 범위에서 튜닝 예정. (LSF-IDM 논문에는 미명시, 일반적인 KD 연구 권장치 따름).

## 4. 아키텍처 리스크 분석 및 개선 제안

| 리스크 번호 | 리스크 설명 | 영향도 | 발생 가능성 | 대응 및 완화 방안 |
|---|---|---|---|---|
| R-01 | 지식 전달의 비효율성 | High | Medium | 1. 지식 증류 하이퍼파라미터(α, T)를 실험적으로 최적화. 2. BiLSTM 외 GRU 등 다른 경량 아키텍처 병행 실험 고려. |
| R-02 | 타겟 환경 제약 조건 미충족 | High | Low | 1. **1.4. 목표 시스템 제약 조건**을 설계의 hard-constraint로 준수. 2. 필요시, 모델 양자화(Quantization) 기법을 추가 적용하여 경량화 목표 달성. |
| R-03 | 데이터셋 과적합(Overfitting) | Medium | Medium | 1. ROAD 데이터셋을 사용한 주기적인 검증으로 일반화 성능 모니터링. 2. 교차 검증(Cross-validation)을 도입하여 모델의 안정성 평가. |

## 5. 데이터셋 및 산출물 관리

- **5.1. 데이터셋**:
    - `data/raw/`: 원본 데이터셋
        - **can_mirgu/**: 교사 모델 사전 훈련 및 미세 조정용 (Training)
            - Benign/: 6일치 정상 주행 데이터 (Day_1 ~ Day_6)
            - Attack/: 3가지 유형의 공격 데이터 (Masquerade, Real, Suspension)
        - **ROAD/**: 모델 검증용 (Validation)
            - ambient/: 정상 주행 시나리오
            - attacks/: 다양한 공격 시나리오
        - **can-train-and-test/**: 최종 성능 평가용 (Test)
            - set_01 ~ set_04: 독립적인 테스트 세트
    - `data/processed/`: 토큰화 및 시퀀싱이 완료된 훈련/검증/테스트용 데이터 파일 (.pkl 또는 .pt 형식)

- **5.2. 산출물 (Artifacts)**:
    - `artifacts/models/`: 훈련된 모델 가중치 파일 (teacher_model.pt, student_model.pt)
    - `artifacts/results/`: 모델 평가 결과 (F1, Latency 등)를 담은 results.csv 파일 및 시각화 자료

## 6. 실험 프로토콜 및 평가 지표

- **6.1. 실험 프로토콜**:
    - **Phase 1**: 데이터 전처리 및 토크나이저 구축
        - can_mirgu 데이터셋에서 어휘집 생성
        - 각 데이터셋별 토큰화 및 시퀀싱 처리
    - **Phase 2**: 교사 모델 사전 훈련 및 미세 조정
        - 사전 훈련: can_mirgu/Benign 데이터 사용 (MLM 태스크)
        - 미세 조정: can_mirgu/Attack 데이터 추가 사용 (분류 태스크)
        - 검증: ROAD 데이터셋으로 베이스라인 성능 확립
    - **Phase 3**: 학생 모델 지식 증류
        - 훈련: can_mirgu 데이터로 교사-학생 지식 전달
        - 검증: ROAD 데이터셋으로 증류 성능 모니터링
        - 하이퍼파라미터 튜닝: α와 T를 최적화
    - **Phase 4**: 최종 평가
        - 테스트: can-train-and-test 데이터셋으로 최종 일반화 성능 측정
        - 비교 분석: 교사 vs 학생 모델의 성능, 효율성 비교

- **6.2. 핵심 평가 지표**:
    - **성능**: F1-Score, Precision, Recall (특히 Fuzzy 공격 유형에 대해)
    - **효율**: 추론 시간(ms/sequence), 모델 파라미터 수, 모델 파일 크기(MB), 추론 시 RAM 사용량(MB)

## 7. 주요 하이퍼파라미터 명세
이 섹션은 프로젝트의 모든 핵심 실험에 대한 파라미터 기준값을 정의합니다.

### 7.1. 교사 모델 사전 훈련 (Pre-training)
- **참고**: `CAN-BERT do it? (Alkhatib et al., 2022)`

| 구분 | 파라미터 | 값 | 근거 / 출처 |
| :--- | :--- | :--- | :--- |
| **모델 아키텍처** | `num_hidden_layers` | `4` | CAN-BERT 논문 Table II |
| | `hidden_size` | `256` | CAN-BERT 논문 Table II |
| | `intermediate_size` | `512` | CAN-BERT 논문 Table II |
| | `num_attention_heads` | `1` | CAN-BERT 논문 Table II |
| | `dropout_prob` | `0.1` | CAN-BERT 논문 Table II |
| **훈련 파라미터** | `mask_prob` | `0.45` | CAN-BERT 논문 Table II |
| | `learning_rate` | `1e-3` | CAN-BERT 논문 Table II |
| | `batch_size` | `32` (GPU당) | CAN-BERT 논문 Table II |
| | `epochs` | `3` (데이터 파트당) | 대규모 데이터셋 점진적 학습 |
| | `optimizer` | `Adam` | CAN-BERT 논문 Table II |

### 7.2. 교사 모델 미세 조정 (Fine-tuning)
- **참고**: `LSF-IDM (Cheng et al., 2023)` 및 차등 학습률 전략

| 파라미터 | 값 | 근거 / 출처 |
| :--- | :--- | :--- |
| `epochs` | `3` | LSF-IDM 논문 Table 4 |
| `batch_size` | `128` | LSF-IDM 논문 Table 4 |
| `body_lr` | `2e-6` | 차등 학습률: 사전 훈련된 BERT 몸통 |
| `head_lr` | `5e-5` | 차등 학습률: 신규 추가된 분류 헤드 |

### 7.3. 학생 모델 지식 증류 (Knowledge Distillation)
- **참고**: `LSF-IDM (Cheng et al., 2023)`

| 구분 | 파라미터 | 값 | 근거 / 출처 |
| :--- | :--- | :--- | :--- |
| **모델 아키텍처** | `hidden_size` | `64` | LSF-IDM 논문 Table 4 |
| | `LSTM layer` | `2` | LSF-IDM 논문 Table 4 |
| | `dropout` | `0.2` | 과적합 방지를 위한 표준 설정 |
| **훈련 파라미터** | `learning_rate` | `1e-5` | LSF-IDM 논문 Table 4 |
| | `batch_size` | `1024` | LSF-IDM 논문 Table 4 |
| | `epochs` | `8` | LSF-IDM 논문 Table 4 |
| | `α (Alpha)` | 초기값: `0.5` <br> 튜닝 범위: `[0.3, 0.7]` | 본 문서 기반 실험적 설정 |
| | `T (Temperature)` | 초기값: `2` <br> 튜닝 범위: `[2, 5]` | 본 문서 기반 실험적 설정 |

## 8. 디렉토리 구조

UGRP/
├── artifacts/
│   ├── models/
│   │   ├── vocab.json 
│   │   └── teacher-model.pt
│   └── results/
│       └── evaluation_report.csv
├── core/
│   ├── __init__.py
│   ├── dataset.py
│   └── tokenizer.py
├── data/
│   ├── processed/
│   └── raw/
│       ├── can_mirgu/              # 교사 모델 사전 훈련용 (Training)
│       │   ├── Attack/             # 공격 데이터
│       │   │   ├── Attacks_metadata.json
│       │   │   ├── Masquerade_attacks/
│       │   │   ├── Real_attacks/
│       │   │   └── Suspension_attacks/
│       │   └── Benign/             # 정상 주행 데이터 (6일치)
│       │       ├── Day_1/
│       │       ├── Day_2/
│       │       ├── Day_3/
│       │       ├── Day_4/
│       │       ├── Day_5/
│       │       └── Day_6/
│       ├── ROAD/                   # 교사/학생 모델 검증용 (Validation)
│       │   ├── ambient/            # 정상 주행 데이터
│       │   ├── attacks/            # 공격 시나리오 데이터
│       │   ├── signal_extractions/ # 신호 추출 데이터
│       │   ├── data_table.csv      # 데이터셋 메타정보
│       │   └── readme.md           # ROAD 데이터셋 설명서
│       └── can-train-and-test/     # 최종 모델 테스트용 (Test)
│           ├── set_01/
│           ├── set_02/
│           ├── set_03/
│           ├── set_04/
│           └── README.md
├── models/
│   ├── __init__.py
│   ├── student.py
│   └── teacher.py
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   └── 2_result_analysis.ipynb
├── scripts/
│   ├── distill.py
│   ├── prepare_dataset.py
│   └── run_full_training.py
├── tests/
│   ├── test_dataset.py
│   ├── test_student.py
│   └── test_model_forward.py
├── .env.example
├── .gitignore
├── requirements.txt
├── specification.md
└── README.md


## 9. 코딩 컨벤션

- **네이밍**: snake_case (변수, 함수), PascalCase (클래스)
- **포맷팅**: [PEP 8](https://www.python.org/dev/peps/pep-0008/) 스타일 가이드 준수. black, isort 도구 사용 권장.
- **주석**: 복잡한 로직이나 설계 의도를 설명하는 주석 필수. Type Hinting (typing 모듈) 적극 사용.

## 10. Git 브랜치 전략

- **전략**: Git-Flow
- **브랜치**:
    - `main`: 배포 가능한 안정 버전.
    - `develop`: 개발의 중심이 되는 브랜치. 모든 기능 브랜치가 이곳으로 병합됨.
    - `feature/<feature-name>`: 신규 기능 개발 브랜치. (e.g., feature/implement-student-model)

## 11. 핵심 기능 사용자 스토리 (User Stories)

### 스토리 1: 교사 모델 훈련

**As a** 연구원,  
**I want to** CAN-BERT 교사 모델을 CAN-MIRGU 데이터셋으로 훈련시켜,  
**So that** 학생 모델이 배워야 할 지식의 성능 기준점을 확립할 수 있다.

**완료 조건(AC)**:
- [x] run_full_training.py 스크립트 실행이 성공적으로 완료됨.
- [x] 훈련된 교사 모델 가중치(teacher_model.pt)가 artifacts/models에 저장됨.
- [x] ROAD 데이터셋에 대한 검증 F1-Score가 목표치(e.g., 0.95 이상)에 도달함.

### 스토리 2: 학생 모델 지식 증류

**As a** 시스템,  
**I want to** 교사 모델의 Soft Label과 실제 정답(Hard Label)을 모두 사용하여 BiLSTM 학생 모델을 훈련하고,  
**So that** 교사의 성능은 유지하면서 ECU 환경에 적합한 경량 모델을 만들 수 있다.

**완료 조건(AC)**:
- [ ] distill.py 스크립트 실행이 성공적으로 완료됨.
- [ ] 지식 증류된 학생 모델(student_model.pt)이 artifacts/models에 저장됨.
- [ ] 학생 모델의 Fuzzy 공격 F1-Score가 교사 모델 대비 5% 이내 하락으로 방어됨.
- [ ] 학생 모델의 추론 속도 및 모델 크기가 KSF(1.2) 목표를 만족함.

## 12. 단계별 작업 계획 (Action Plan)

### Milestone 1: Phase 2 - 교사 모델 베이스라인 확립 (완료)

- [x] **Task**: run_full_training.py 스크립트 완성
- [x] **Task**: CAN-MIRGU 전체 데이터셋 훈련 및 teacher_model.pt 확보

### Milestone 2: Phase 3 - 학생 모델 구현 및 지식 증류

- [x] **Task**: models/student.py에 BiLSTM 아키텍처 구현 (우선순위: High, 예상 소요: 4h)
- [ ] **Task**: scripts/distill.py에 지식 증류 훈련 로직 및 복합 손실 함수 구현 (우선순위: High, 예상 소요: 8h)
- [ ] **Task**: 초기 하이퍼파라미터(α=0.5, T=2)로 1차 지식 증류 실행 및 성능 측정 (우선순위: High, 예상 소요: 6h)
- [ ] **Task**: 하이퍼파라미터 튜닝 실험 (α, T 값 변경) (우선순위: Medium, 예상 소요: 12h)

### Milestone 3: Phase 4 - 최종 평가 및 결과 정리

- [ ] **Task**: can-train-and-test 데이터셋으로 최종 모델 일반화 성능 평가 (우선순위: High, 예상 소요: 4h)
- [ ] **Task**: 교사 vs. 학생 모델 성능/효율 비교 분석 보고서 작성 (우선순위: High, 예상 소요: 8h)
- [ ] **Task**: 프로젝트 최종 발표 자료 및 README 문서 정리 (우선순위: Medium, 예상 소요: 6h)

## 13. 설치 및 실행 방법

### 의존성 설치:
pip install -r requirements.txt

### 환경변수 설정:
.env.example 파일을 .env로 복사 후, 필요한 환경 변수(e.g., 데이터 경로)를 설정합니다.

### 실행:
모든 Python 스크립트는 패키지 루트에서 -m 옵션으로 실행하는 것을 원칙으로 합니다.

### 교사 모델 훈련 실행 예시:
python -m scripts.run_full_training

### 지식 증류 실행 예시:
python -m scripts.distill