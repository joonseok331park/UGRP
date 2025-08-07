# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CAN-IDS (Controller Area Network - Intrusion Detection System) project that implements a BERT-based teacher model for anomaly detection in CAN bus data, with knowledge distillation to a lightweight BiLSTM student model for deployment in resource-constrained ECU environments.

## Architecture

The project follows a teacher-student knowledge distillation approach:

### Core Components
- **`core/tokenizer.py`**: Contains `CANTokenizer` and `CANSequencer` classes
  - `CANTokenizer`: Manages vocabulary for CAN data with offset-based token mapping (ID_OFFSET = 260)
  - `CANSequencer`: Transforms CAN data into fixed-length sequences for model input
  - Uses special tokens: `<PAD>`, `<UNK>`, `<MASK>`, `<VOID>`

- **`core/dataset.py`**: Contains `MLMDataset` class for BERT-style masked language modeling
  - Implements dynamic masking with 15% probability following BERT standards
  - 80% mask token, 10% random token, 10% original token

### Models
- **`models/teacher.py`**: Contains `CANBertForMaskedLM` teacher model
  - Built using Hugging Face Transformers with custom BertConfig
  - Default architecture: 256 hidden size, 4 layers, variable attention heads

- **`models/student.py`**: Contains BiLSTM student model (placeholder, needs implementation)
  - Target architecture: 2-layer stacked BiLSTM with 64 hidden units
  - Designed for real-time inference with <0.1ms latency per sequence

### Data Processing
- **`utils/data_loader.py`**: Contains `load_can_data()` function
  - Parses HCRL dataset format (timestamp ID#DLC data label)
  - Standardizes CAN messages to 8-byte data fields with padding
  - Returns DataFrame with columns: Timestamp, CAN_ID, DLC, Data, Label

### Training Scripts
- **`scripts/pretrain.py`**: Main pre-training script for teacher model
  - Supports resuming from checkpoints
  - Uses Weights & Biases (wandb) for logging
  - Implements AdamW optimizer with linear warmup scheduling
  - Gradient clipping and mixed precision training ready

- **`scripts/aggregate_data.py`**: Data aggregation utility
  - Merges multiple .log files from CAN-MIRGU dataset structure
  - Creates consolidated training data files

- **`scripts/distill.py`**: Knowledge distillation script (placeholder, needs implementation)
  - Will implement teacher-student distillation with combined loss (α * L_CE + (1-α) * L_KD)

## Development Commands

### Data Preparation
```bash
# Aggregate CAN-MIRGU dataset files
python scripts/aggregate_data.py
```

### Running Tests
Each module includes test code in the `if __name__ == '__main__':` block. Run individual modules directly:
```bash
python core/tokenizer.py
python core/dataset.py  
python models/teacher.py
python utils/data_loader.py
```

### Pre-training Teacher Model
```bash
python scripts/pretrain.py \
  --data_path /path/to/data.log \
  --vocab_path /path/to/vocab.json \
  --output_dir /path/to/checkpoints \
  --seq_len 126 \
  --batch_size 64 \
  --epochs 20 \
  --learning_rate 5e-5
```

### Resume Training
```bash
python scripts/pretrain.py \
  --resume_from_checkpoint /path/to/checkpoint.pt \
  [other arguments...]
```

### Knowledge Distillation (Coming Soon)
```bash
python scripts/distill.py \
  --teacher_checkpoint /path/to/teacher.pt \
  --data_path /path/to/data.log \
  --output_dir /path/to/student_checkpoints \
  --alpha 0.5 \
  --temperature 2 \
  --batch_size 1024 \
  --epochs 8
```

## Key Technical Details

### Model Specifications
- **Teacher Model**: BERT-based with 4 layers, 256 hidden size, ~10MB model size
- **Student Model**: 2-layer BiLSTM, 64 hidden size, target <16MB model size
- **Sequence Length**: Default 126 tokens (configurable)
- **Vocabulary**: Offset-based with data tokens (00-FF) and ID tokens (offset by 260)

### Performance Targets
- **Teacher F1-Score**: >0.95 on ROAD dataset
- **Student F1-Score**: Within 5% of teacher performance
- **Student Inference Time**: <0.1ms per sequence
- **Student Memory Usage**: <128MB RAM

### Data Format
- **Input**: HCRL format with timestamp, CAN ID, DLC, data payload, and label
- **Processing**: Standardizes to 8-byte data fields with padding
- **Output**: Tokenized sequences ready for model input

## Key Hyperparameters

### Teacher Pre-training (from CAN-BERT paper)
- Learning rate: 1e-3
- Mask probability: 0.45
- Batch size: 32 per GPU
- Epochs: 3 per data partition

### Knowledge Distillation (from LSF-IDM paper)
- Learning rate: 1e-5
- Batch size: 1024
- Epochs: 8
- Alpha (α): 0.5 (tunable in [0.3, 0.7])
- Temperature (T): 2 (tunable in [2, 5])

## Dependencies

Main dependencies from requirements.txt:
- PyTorch (latest stable version for Colab)
- Transformers (Hugging Face)
- pandas
- scikit-learn
- tensorboard
- wandb (for experiment tracking)
- tqdm (for progress bars)

## Project Status

### Completed
- Teacher model architecture and pre-training script
- Core tokenization and dataset modules
- Data loading utilities

### In Progress
- Student model implementation (models/student.py)
- Knowledge distillation script (scripts/distill.py)
- Comprehensive evaluation pipeline

### TODO
- Implement BiLSTM student model architecture
- Create distillation training loop with combined loss
- Add model quantization for further size reduction
- Set up comprehensive evaluation on can-train-and-test dataset

## Cognitive Processing Strategies

### Problem-Solving Approaches
- 복잡한 문제를 해결할 때는 항상 Sequential Thinking MCP 서버를 사용하여 체계적으로 접근한다. 이 도구는 문제를 단계별로 분해하고, 필요시 이전 단계를 수정하거나 대안을 탐색할 수 있게 해준다.

## Server Usage Guidelines

### 1. filesystem MCP 서버

**메모리 저장 지침:**

"파일 시스템 작업(읽기, 쓰기, 디렉토리 탐색)을 할 때는 filesystem MCP 서버를 우선적으로 사용한다. 특히 다음 상황에서 활용한다:

- 디렉토리 구조를 JSON 트리로 확인할 때: mcp__filesystem__directory_tree

- 여러 파일을 동시에 읽을 때: mcp__filesystem__read_multiple_files

- 파일 크기 정보가 필요한 디렉토리 목록: mcp__filesystem__list_directory_with_sizes

- 파일 검색 시: mcp__filesystem__search_files

- 파일 메타데이터 확인: mcp__filesystem__get_file_info"

### 2. desktop-commander MCP 서버

**메모리 저장 지침:**

"시스템 명령 실행, 프로세스 관리, 코드 검색을 할 때는 desktop-commander MCP 서버를 사용한다. 특히 다음 상황에서 활용한다:

- CSV, JSON 등 로컬 파일 분석 시: mcp__desktop-commander__start_process + interact_with_process (Python REPL)

- 코드 패턴 검색 시: mcp__desktop-commander__search_code (ripgrep 기반)

- 프로세스 관리가 필요할 때: list_sessions, force_terminate

- 파일 블록 단위 편집: mcp__desktop-commander__edit_block

- 설정 관리: get_config, set_config_value"

### 3. context7 MCP 서버

**메모리 저장 지침:**

"라이브러리나 프레임워크의 최신 문서가 필요할 때는 context7 MCP 서버를 사용한다:

- 라이브러리 이름으로 ID 검색: mcp__context7__resolve-library-id

- 문서 가져오기: mcp__context7__get-library-docs

- MongoDB, Next.js, Supabase 등 주요 라이브러리 문서 접근 시 활용"

### 4. exa MCP 서버

**메모리 저장 지침:**

"최신 정보나 웹 검색이 필요할 때는 exa MCP 서버를 사용한다:

- 일반 웹 검색: mcp__exa__web_search_exa

- 기업 정보 조사: mcp__exa__company_research_exa

- 특정 URL 콘텐츠 추출: mcp__exa__crawling_exa

- LinkedIn 프로필/회사 검색: mcp__exa__linkedin_search_exa

- 복잡한 연구 주제: mcp__exa__deep_researcher_start/check (AI 에이전트가 심층 조사)"

## MCP 서버 선택 우선순위



### 파일 작업

1순위: filesystem MCP (전문화된 파일 도구)

2순위: desktop-commander MCP (추가 기능 제공)

3순위: 기본 Read/Write/Edit 도구



### 시스템 작업

1순위: desktop-commander MCP (프로세스 관리, 대화형 REPL)

2순위: Bash 도구 (단순 명령)



### 코드/텍스트 검색

1순위: desktop-commander__search_code (ripgrep 기반, 정규식 지원)

2순위: filesystem__search_files (파일명 검색)

3순위: Grep 도구



### 데이터 분석 (CSV, JSON 등)

1순위: desktop-commander MCP의 start_process + interact_with_process (Python/Node REPL)

2순위: 직접 파일 읽기 후 처리



### 문서/정보 검색

1순위: context7 MCP (라이브러리 문서)

2순위: exa MCP (웹 검색, 최신 정보)

3순위: WebSearch/WebFetch 도구



### 복잡한 문제 해결

1순위: sequentialthinking MCP (체계적 사고 과정)

2순위: Task 도구 (에이전트 활용)



## 주의사항

- 각 MCP 서버는 특화된 기능을 제공하므로, 작업에 가장 적합한 서버를 선택한다

- 동일한 기능이 여러 서버에 있을 경우, 더 전문화된 서버를 우선 사용한다

- MCP 서버가 실패할 경우 대체 도구를 사용한다

## Memories

### PCIP Framework: Dynamic Expert System
- Implemented a multi-layered adaptive expertise management framework
- Supports dynamic expert selection based on conversational context
- Integrates external knowledge sources for enhanced problem-solving
- Provides risk-based execution modes (Silent vs Explicit)
- Enables seamless transitions between domain experts during task execution

### Git Workflow Memory
- Implemented a comprehensive Git workflow guide for collaborative development
- Emphasizes creating feature branches from develop
- Provides clear guidelines for committing, pushing, and creating pull requests
- Highlights the importance of clean, descriptive commit messages
- Includes best practices for branch management and merging