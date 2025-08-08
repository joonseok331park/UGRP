# UGRP/scripts/run_full_training.py

import logging
import argparse
from pathlib import Path
import pandas as pd

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from core.tokenizer import CANTokenizer
from core.dataset import MLMDataset, _load_and_parse_log
from models.teacher import CANBert

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pretraining(args):
    """교사 모델의 MLM 사전 훈련을 수행하는 메인 함수"""
    
    logging.info("="*50)
    logging.info("Phase 2: 교사 모델 MLM 사전 훈련을 시작합니다.")
    logging.info("="*50)

    # 1. 경로 설정 (Pathlib 사용)
    processed_data_path = Path(args.data_path)
    vocab_path = Path(args.vocab_path)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. 토크나이저 준비
    logging.info("토크나이저를 준비합니다...")
    tokenizer = CANTokenizer()
    if vocab_path.exists():
        logging.info(f"기존 어휘집 파일을 로드합니다: {vocab_path}")
        tokenizer.load_vocab(str(vocab_path))
    else:
        logging.info(f"새 어휘집을 생성합니다. 소스: {processed_data_path}")
        # 어휘집 생성을 위해 데이터 파싱 (메모리 효율성을 위해 일부만 사용할 수도 있음)
        can_df = _load_and_parse_log(str(processed_data_path))
        tokenizer.build_vocab(can_df)
        tokenizer.save_vocab(str(vocab_path))
        logging.info(f"새 어휘집을 '{vocab_path}'에 저장했습니다.")
    
    vocab_size = tokenizer.get_vocab_size()
    logging.info(f"어휘집 크기: {vocab_size}")

    # 3. 데이터셋 및 데이터 콜레이터 준비
    logging.info("MLM 데이터셋을 준비합니다...")
    train_dataset = MLMDataset(
        file_path=str(processed_data_path),
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        mask_prob=args.mask_prob
    )
    
    # 데이터 콜레이터: 배치 생성 및 자동 마스킹 담당
    # specification.md 7.1. (mask_prob=0.45)를 따르지만, args로 조절 가능
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=None, # 우리 커스텀 토크나이저 대신 직접 마스킹하므로 None
        mlm=True,
        mlm_probability=args.mask_prob
    )
    # MLMDataset이 이미 마스킹을 처리하므로, Hugging Face collator의 마스킹은 비활성화합니다.
    # 우리 Dataset의 __getitem__이 딕셔너리를 반환하므로 collator는 단순 배치화 역할만 합니다.
    # 만약 Dataset에서 마스킹을 안한다면, collator에서 처리하도록 tokenizer와 mlm_probability를 설정해야 합니다.
    # 현재 MLMDataset 구현이 더 정교하므로 collator는 기본 PyTorch collator처럼 동작하게 둡니다.
    # (HuggingFace Trainer는 DataCollator를 필수로 요구하므로, 형태만 맞춰줍니다)
    # -> 더 나은 방법: Dataset에서 마스킹 로직을 제거하고 DataCollator에 위임
    # -> 현재 코드: Dataset의 정교한 마스킹 로직을 존중하여 그대로 사용
    # -> Trainer는 딕셔너리 리스트를 받아 자동으로 텐서 배치를 만듭니다.

    logging.info(f"데이터셋 샘플 수: {len(train_dataset):,}")

    # 4. 모델 준비
    logging.info("CAN-BERT 교사 모델을 로드합니다...")
    model = CANBert.from_spec(vocab_size=vocab_size)
    logging.info(f"모델 파라미터 수: {model.num_parameters():,}")

    # 5. 훈련 설정 (TrainingArguments)
    # specification.md 7.1.의 훈련 파라미터를 정확히 반영
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_total_limit=2,
        save_steps=5000, # 5000 스텝마다 체크포인트 저장
        logging_steps=500,  # 500 스텝마다 로그 출력
        report_to="none",   # wandb/tensorboard 연동 비활성화
        fp16=torch.cuda.is_available(), # GPU 사용 가능 시 혼합 정밀도 훈련 사용
    )

    # 6. Trainer 초기화 및 훈련 시작
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # data_collator는 기본값 사용 (딕셔너리 배칭)
    )

    logging.info("사전 훈련을 시작합니다...")
    trainer.train()
    logging.info("사전 훈련이 성공적으로 완료되었습니다.")

    # 7. 최종 모델 및 토크나이저 저장
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_vocab(str(final_model_path / "vocab.json"))
    logging.info(f"훈련된 최종 모델과 어휘집이 '{final_model_path}'에 저장되었습니다.")
    logging.info("="*50)
    logging.info("Phase 2 완료!")
    logging.info("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CAN-BERT Teacher Model Pre-training Script")
    
    # 경로 관련 인자
    parser.add_argument("--data_path", type=str, default="data/processed/can_mirgu_benign.log", help="훈련 데이터 파일 경로")
    parser.add_argument("--vocab_path", type=str, default="artifacts/models/can_vocab.json", help="어휘집 파일 경로")
    parser.add_argument("--output_dir", type=str, default="artifacts/models/teacher_model", help="훈련된 모델과 체크포인트 저장 경로")
    
    # 훈련 하이퍼파라미터 (specification.md 7.1. 기반)
    parser.add_argument("--seq_len", type=int, default=126, help="입력 시퀀스 길이")
    parser.add_argument("--mask_prob", type=float, default=0.45, help="MLM 마스킹 확률")
    parser.add_argument("--epochs", type=int, default=3, help="총 훈련 에포크")
    parser.add_argument("--batch_size", type=int, default=32, help="훈련 배치 사이즈")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="학습률")

    args = parser.parse_args()
    run_pretraining(args)