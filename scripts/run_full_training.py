# UGRP/scripts/run_full_training.py (v2.1)

import logging
import argparse
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_utils import set_seed

# [수정] MLMDataset은 이제 마스킹을 책임지지 않으므로, 더 범용적인 이름의 Dataset 클래스를 사용합니다.
# ClassificationDataset을 MLM 훈련에도 재사용할 수 있도록 수정합니다.
from core.tokenizer import CANTokenizer
from core.dataset import CANClassificationDataset, _load_and_parse_log 
from models.teacher import CANBert

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    """교사 모델의 MLM 사전 훈련을 수행하는 메인 함수 (v2.1)"""
    
    logging.info("="*50)
    logging.info("Phase 2: 교사 모델 MLM 사전 훈련 (v2.1)을 시작합니다.")
    logging.info(f"훈련 목표: CAN-BERT (2022) 논문 재현")
    logging.info("="*50)

    set_seed(args.seed)
    logging.info(f"Random seed 고정: {args.seed}")

    # 경로 설정
    processed_data_path = Path(args.data_path)
    vocab_path = Path(args.vocab_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 토크나이저 준비
    tokenizer = CANTokenizer()
    # 어휘집 파일이 없으면 훈련 데이터로 새로 생성
    if not vocab_path.exists():
        logging.warning(f"어휘집 파일({vocab_path})이 없어 새로 생성합니다.")
        can_df = _load_and_parse_log(str(processed_data_path))
        tokenizer.build_vocab(can_df)
        tokenizer.save_vocab(str(vocab_path))
        logging.info(f"새 어휘집을 '{vocab_path}'에 저장했습니다.")
    else:
        tokenizer.load_vocab(str(vocab_path))
    
    vocab_size = tokenizer.get_vocab_size()
    logging.info(f"어휘집 로드 완료. 크기: {vocab_size}")

    # 데이터셋 준비 (이제 마스킹 로직 없음)
    # CANClassificationDataset은 라벨을 생성하지만, MLM에서는 사용되지 않음.
    full_dataset = CANClassificationDataset(
        file_path=str(processed_data_path),
        tokenizer=tokenizer,
        seq_len=args.seq_len
    )
    
    train_size = int(0.9 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])
    logging.info(f"데이터셋 분할 완료: 훈련 {len(train_dataset):,}개, 검증 {len(eval_dataset):,}개")

    # 데이터 콜레이터 (마스킹 책임 담당)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=None, # HF Tokenizer 객체가 아니므로 None
        mlm=True,
        mlm_probability=args.mask_prob,
    )
    
    # 모델 준비
    model = CANBert.from_spec(vocab_size=vocab_size)
    
    # 훈련 설정 (하이퍼파라미터 기본값을 논문 기준으로 수정)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        
        evaluation_strategy="steps",
        eval_steps=2000,
        logging_steps=500,
        report_to="tensorboard",

        save_strategy="steps",
        save_steps=2000,
        save_total_limit=3,
        load_best_model_at_end=True,

        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logging.info("사전 훈련을 시작합니다...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    trainer.save_model()
    tokenizer.save_vocab(str(Path(args.output_dir) / "vocab.json"))
    logging.info(f"훈련 완료! 최적 모델이 '{args.output_dir}'에 저장되었습니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CAN-BERT Teacher Model Pre-training Script (v2.1)")
    
    parser.add_argument("--data_path", type=str, default="data/processed/can_mirgu_benign.log")
    parser.add_argument("--vocab_path", type=str, default="artifacts/models/vocab.json")
    parser.add_argument("--output_dir", type=str, default="artifacts/models/teacher_model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # 하이퍼파라미터 기본값을 CAN-BERT 논문 및 specification.md 기준으로 설정
    parser.add_argument("--seq_len", type=int, default=126)
    parser.add_argument("--mask_prob", type=float, default=0.45)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=0) # CAN-BERT 논문에는 warmup이 명시되지 않음
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)