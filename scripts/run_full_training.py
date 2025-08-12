# UGRP/scripts/run_full_training.py (v2.2 최종)

import logging
import argparse
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_utils import set_seed

from core.tokenizer import CANTokenizer
from core.dataset import ClassificationDataset, _load_and_parse_log 
from models.teacher import CANBert

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    """교사 모델의 MLM 사전 훈련을 수행하는 메인 함수 (v2.2 최종)"""
    
    logging.info("="*50)
    logging.info("Phase 2: 교사 모델 MLM 사전 훈련 (v2.2)을 시작합니다.")
    logging.info(f"훈련 목표: CAN-BERT (2022) 논문 재현 (최적화 적용)")
    logging.info("="*50)

    set_seed(args.seed)
    logging.info(f"Random seed 고정: {args.seed}")

    processed_data_path = Path(args.data_path)
    vocab_path = Path(args.vocab_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CANTokenizer()
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

    # [수정] 데이터셋 준비 로직 전체를 캐싱 기능으로 감쌉니다.
    cached_dataset_path = processed_data_path.parent / f"cached_dataset_seq{args.seq_len}_stride{args.stride}.pt"
    
    if cached_dataset_path.exists() and not args.force_reload:
        logging.info(f"캐시된 데이터셋을 로드합니다: {cached_dataset_path}")
        full_dataset = torch.load(cached_dataset_path)
    else:
        logging.info("새로운 데이터셋을 생성하고 캐싱합니다...")
        full_dataset = ClassificationDataset(
            file_path=str(processed_data_path),
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride
        )
        torch.save(full_dataset, cached_dataset_path)
        logging.info(f"데이터셋을 캐시 파일로 저장했습니다: {cached_dataset_path}")
    
    train_size = int(0.9 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])
    logging.info(f"데이터셋 분할 완료: 훈련 {len(train_dataset):,}개, 검증 {len(eval_dataset):,}개")

    data_collator = DataCollatorForLanguageModeling(
        # [수정] tokenizer=None -> tokenizer=tokenizer
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mask_prob,
    )
    
    model = CANBert.from_spec(vocab_size=vocab_size)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        
        # [추가] 그래디언트 축적 설정
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        evaluation_strategy="steps",
        eval_steps=1000, # 검증 스텝 조정
        logging_steps=500,
        report_to="tensorboard",

        save_strategy="steps",
        save_steps=1000, # 저장 스텝 조정
        save_total_limit=3,
        load_best_model_at_end=True,

        fp16=torch.cuda.is_available(),
        dataloader_num_workers=args.num_workers,
        
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
    parser = argparse.ArgumentParser(description="CAN-BERT Teacher Model Pre-training Script (v2.2)")
    
    parser.add_argument("--data_path", type=str, default="data/processed/can_mirgu_benign.log")
    parser.add_argument("--vocab_path", type=str, default="artifacts/models/vocab.json")
    parser.add_argument("--output_dir", type=str, default="artifacts/models/teacher_model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--force_reload", action='store_true', help="데이터셋 캐시를 무시하고 새로 로드")

    # 논문 재현 하이퍼파라미터
    parser.add_argument("--seq_len", type=int, default=126)
    parser.add_argument("--mask_prob", type=float, default=0.45)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # [추가] 성능 최적화 인자
    parser.add_argument("--stride", type=int, default=63, help="데이터셋 생성 시 슬라이딩 윈도우 스트라이드")
    parser.add_argument("--num_workers", type=int, default=4, help="데이터 로더 워커 수")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="그래디언트 축적 스텝")

    args = parser.parse_args()
    main(args)