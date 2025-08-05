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