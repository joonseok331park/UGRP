# Repository Guidelines

## Project Overview
- CAN-IDS 프로젝트: BERT 기반 교사(CANBert) 지식을 경량 BiLSTM 학생 모델에 증류하여, 자원 제약 ECU 환경에서 실시간 침입 탐지를 목표로 합니다.

## Project Structure & Modules
- `core/`: Tokenization and datasets (`core/tokenizer.py`, `core/dataset.py`).
- `models/`: Model code (`student.py` BiLSTM, `teacher.py` CANBert).
- `scripts/`: Utilities and training (`pretrain.py`, `aggregate_data.py`, `distill.py` placeholder).
- `tests/`: Test files (e.g., `tests/test_student.py`; add new tests as `tests/test_<module>.py`).
- `data/`: Local datasets (`raw/`, `processed/`). Keep large/raw data out of git.
- `artifacts/`: Checkpoints and results (`artifacts/models/`, `artifacts/results/`).
- `notebooks/`: Exploration and analysis notebooks.

## Architecture Overview
- Pipeline: tokenize/sequence CAN logs → pre-train teacher (MLM) → fine-tune (if needed) → distill to BiLSTM student → evaluate/deploy.
- Tokenizer: `CANTokenizer` with ID offset 256 and special tokens (`<PAD>`, `<UNK>`, `<MASK>`, `<VOID>`); sequences built via sliding window (default `seq_len=126`).
- Datasets: `MLMDataset` for teacher pre-training (masked tokens), `ClassificationDataset` for downstream/KD.
- Models: Teacher `CANBert` (BERTForMaskedLM: 4 layers, hidden 256, intermed 512); Student `BiLSTM` (2 layers, hidden 64, dropout 0.2).
- Distillation: combined loss `L = α·CE + (1−α)·KD` with temperature `T` (see `scripts/distill.py`).
- Targets: <0.1ms/sequence inference (student), <16MB model size, <128MB RAM.

## Key Hyperparameters & Targets (Refined)
- Teacher pre-train: `seq_len=126`, `batch_size=32`, `learning_rate=5e-5`, `warmup_steps=1000`, MLM mask prob `0.15`.
- Teacher config: hidden `256`, layers `4`, heads `1`, intermed `512`.
- Student: hidden `64`, layers `2`, dropout `0.2`.
- KD: `alpha=0.5` (tune 0.3–0.7), `temperature=2` (tune 2–5).
- Performance: teacher F1 on ROAD > 0.95; student F1 within 5% of teacher; latency/size/RAM as above.

## Build, Test, and Development Commands
- Setup: `pip install -r requirements.txt` (use Python 3.12; optional venv).
- Aggregate CAN-MIRGU logs: `python scripts/aggregate_data.py` → `data/HCRL_dataset/train_aggregated.log`.
- Pre-train teacher (example):
  `python scripts/pretrain.py --data_path data/HCRL_dataset/train_aggregated.log --vocab_path artifacts/models/vocab.json --output_dir artifacts/models --seq_len 126 --batch_size 32`.
- Resume training:
  `python scripts/pretrain.py --resume_from_checkpoint artifacts/models/<checkpoint>.pt --data_path ... --vocab_path ...`
- Run student model suite: `python tests/test_student.py`.
- Module self-checks: `python core/tokenizer.py`, `python core/dataset.py`, `python models/teacher.py`.
- Distillation (placeholder):
  `python scripts/distill.py --teacher_checkpoint artifacts/models/teacher.pt --data_path data/... --output_dir artifacts/models/student --alpha 0.5 --temperature 2 --batch_size 1024 --epochs 8`.

## Dependencies
- From `requirements.txt`: `torch`, `torchvision`, `torchaudio`, `transformers`, `pandas`, `scikit-learn`, `tensorboard`, `wandb`.
- Also used: `tqdm` (if not installed, `pip install tqdm`).

## Data Format
- Input: HCRL 스타일 `(timestamp) can0 ID#DATA LABEL` 라인 포맷.
- Processing: payload를 8바이트(16 hex)로 패딩, 2자씩 분할; ID는 토크나이저 오프셋(256) 적용; 슬라이딩 윈도우로 시퀀스 생성.
- Output: 토큰 ID 시퀀스와 마스크/레이블(`MLMDataset`은 15% 마스킹: 80% `<MASK>`, 10% 랜덤, 10% 원본 유지).

## Coding Style & Naming Conventions
- Style: PEP 8, 4-space indent, type hints for public APIs, concise docstrings.
- Naming: `snake_case` for files/functions, `PascalCase` for classes; tests as `tests/test_*.py`.
- Formatting: prefer `black` and `isort` locally (before PRs).

## Testing Guidelines
- Scope: cover tokenization (offset logic), dataset windowing/masking, and model forward with variable lengths.
- Conventions: place tests in `tests/`, name files `test_*.py`, use clear assertions and small fixtures.
- Quick checks: run `python tests/test_student.py`; add minimal repros under `tests/` when fixing bugs.

## Commit & Pull Request Guidelines
- Branching: create `feature/<name>` from `develop`; do not commit directly to `develop`.
- Commits: use prefixes `feat:`, `fix:`, `docs:`, `test:`, `refactor:` (e.g., `feat: add BiLSTM student forward`).
- PRs: target `develop`; include description, linked issues/spec sections, run steps (commands), and relevant logs/screenshots; ensure tests and formatters pass; update docs (e.g., `specification.md`) if behavior changes.

## Security & Configuration
- Copy `.env.example` to `.env`; never commit secrets. Keep raw datasets outside version control or under `data/raw/` (ignored). Store checkpoints in `artifacts/` and avoid pushing large binaries.

## Agent-Specific Instructions
- Language: Always respond in Korean for all interactions in this repository.
- Canonical memory: Treat `AGENTS.md` as the source of truth for agent behavior and repo practices.
- Tone: Keep responses concise, professional, and actionable; include concrete commands and paths when helpful.

## Status & TODO (Refined)
- Completed: teacher architecture (`models/teacher.py`), tokenizer/datasets (`core/`), aggregation script, student baseline (`models/student.py`).
- In progress: `scripts/distill.py` (KD training loop and combined loss), broader unit tests.
- Next: align teacher import naming in `scripts/pretrain.py`, finalize KD script and add evaluation pipeline, add optional quantization for further size reduction.

## Notes & Alignment
- ID offset is `256` (update from older notes mentioning 260).
- Prefer `core/dataset.py` for data loading; references to `utils/data_loader.py` in older docs are deprecated.
- Teacher class is `CANBert`; `scripts/pretrain.py` currently imports `CANBertForMaskedLM` (rename or adjust import when updating the script).
