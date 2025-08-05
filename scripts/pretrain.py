# scripts/pretrain.py (모든 오류 수정 및 개선사항이 반영된 최종 버전)

import argparse
import os
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertConfig, get_linear_schedule_with_warmup

from core.tokenizer import CANTokenizer
from core.dataset import MLMDataset
from models.teacher import CANBertForMaskedLM
from utils.data_loader import load_can_data

def setup_logging_and_device(args):
    """wandb를 초기화하고 학습 장치(GPU/CPU)를 설정합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # os.environ["WANDB_MODE"] = "disabled" # wandb를 사용하지 않으려면 이 줄의 주석을 해제하세요.
    wandb.init(
        project="CAN-IDS-Pretraining",
        config=vars(args)
    )
    return device

def prepare_data(args, tokenizer):
    """메모리 효율적인 MLMDataset을 초기화하고 DataLoader를 준비합니다."""
    print("Initializing memory-efficient MLMDataset...")
    
    mlm_dataset = MLMDataset(
        file_path=args.data_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        dataset_type=args.dataset_type
    )
    
    if len(mlm_dataset) == 0:
        raise ValueError("MLM dataset is empty. Check data file or tokenizer.")
    
    train_loader = DataLoader(
        mlm_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    print(f"Data preparation complete. Total trainable sequences: {len(mlm_dataset):,}")
    return train_loader

def prepare_model_and_optimizer(args, model, train_loader):
    """옵티마이저와 학습률 스케줄러를 초기화합니다."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    return optimizer, scheduler

def train_one_epoch(model, data_loader, optimizer, scheduler, device, epoch, epochs):
    """단일 학습 에폭을 실행합니다."""
    model.train()
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 100 == 0:
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        progress_bar.set_postfix({'loss': loss.item()})

def save_checkpoint(model, optimizer, scheduler, epoch, output_dir):
    """학습 체크포인트를 저장합니다."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    checkpoint_path = os.path.join(output_dir, f"can-bert-pretrained-epoch-{epoch+1}.pt")
    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    
    print(f"\nEpoch {epoch+1} | Checkpoint saved to {checkpoint_path}")

def main(args):
    """CAN-BERT 사전 훈련 프로세스를 실행하는 메인 함수."""
    device = setup_logging_and_device(args)
    
    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {args.vocab_path}")
    
    tokenizer = CANTokenizer()
    tokenizer.load_vocab(args.vocab_path)
    
    config = BertConfig(
        vocab_size=len(tokenizer.token_to_id),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=1,
        intermediate_size=512,
        max_position_embeddings=args.seq_len
    )
    model = CANBertForMaskedLM(config)
    model.to(device)
    
    train_loader = prepare_data(args, tokenizer)
    optimizer, scheduler = prepare_model_and_optimizer(args, model, train_loader)
    
    start_epoch = 0
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            print(f"Loading checkpoint '{args.resume_from_checkpoint}'")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found at '{args.resume_from_checkpoint}'")
            return

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, data_loader, optimizer, scheduler, device, epoch, args.epochs)
        save_checkpoint(model, optimizer, scheduler, epoch, args.output_dir)
    
    print("Training finished.")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CAN-BERT Pre-training Script")
    
    # --- 모든 인자가 올바르게 수정되었습니다 ---
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocab.json file.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/", help="Directory to save model checkpoints.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--dataset_type", type=str, default='candump', help="Type of the dataset format (e.g., 'hcrl', 'candump').")
    parser.add_argument("--seq_len", type=int, default=126, help="Sequence length for the model input.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Total number of training epochs for each data part.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate for AdamW.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for the learning rate scheduler.")
    
    args = parser.parse_args()
    main(args)