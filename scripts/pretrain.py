import argparse
import os
import sys
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Add the project root to the Python path to allow for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tokenizer import CANTokenizer, CANSequencer
from core.dataset import MLMDataset
from models.teacher import BertConfig, CANBertForMaskedLM
from utils.data_loader import load_can_data

def setup_logging_and_device(args):
    """Initializes wandb and sets up the device (GPU/CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="CAN-IDS-Pretraining",
        config=vars(args)
    )
    return device

def prepare_data(args, tokenizer):
    """Loads data, creates sequences, and prepares the DataLoader."""
    print("Loading data...")
    can_df = load_can_data(args.data_path)
    
    print("Preparing sequences...")
    sequencer = CANSequencer(tokenizer, seq_len=args.seq_len)
    sequences = sequencer.transform(can_df['can_raw'].tolist())
    
    mlm_dataset = MLMDataset(sequences, tokenizer, seq_len=args.seq_len)
    
    train_loader = DataLoader(
        mlm_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print("Data preparation complete.")
    return train_loader

def prepare_model_and_optimizer(args, model, train_loader):
    """Initializes the optimizer and learning rate scheduler."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    return optimizer, scheduler

def train_one_epoch(model, data_loader, optimizer, scheduler, device, epoch, epochs):
    """Runs a single training epoch."""
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
        
        if step % 10 == 0:
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        progress_bar.set_postfix({'loss': loss.item()})

def save_checkpoint(model, optimizer, scheduler, epoch, output_dir):
    """Saves a training checkpoint."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    checkpoint_path = os.path.join(output_dir, f"can-bert-pretrained-epoch-{epoch+1}.pt")
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Epoch {epoch+1} | Checkpoint saved to {checkpoint_path}")

def main(args):
    """Main function to run the CAN-BERT pre-training process."""
    device = setup_logging_and_device(args)
    
    # Load tokenizer
    tokenizer = CANTokenizer.from_file(args.vocab_path)
    
    # Prepare data
    train_loader = prepare_data(args, tokenizer)
    
    # Initialize model
    config = BertConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=args.seq_len
    )
    model = CANBertForMaskedLM(config)
    model.to(device)
    
    # Prepare optimizer and scheduler
    optimizer, scheduler = prepare_model_and_optimizer(args, model, train_loader)
    
    start_epoch = 0
    
    # Resume from checkpoint if specified
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
            sys.exit(1)

    # Training Loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, args.epochs)
        save_checkpoint(model, optimizer, scheduler, epoch, args.output_dir)
    
    print("Training finished.")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CAN-BERT Pre-training Script")
    
    # Path arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocab.json file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    
    # Model and training arguments
    parser.add_argument("--seq_len", type=int, default=126, help="Sequence length for the model input.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Total number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate for AdamW.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for the learning rate scheduler.")
    
    args = parser.parse_args()
    main(args)