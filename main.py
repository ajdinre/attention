import torch
import argparse

from torch.utils.data import DataLoader

from model.transformer import Transformer
from data.dataset import WMTDataset
from train import train_model
from evaluate import evaluate_model
from utils import get_device, count_parameters, Timer, log_model_summary, visualize_model


def main():
    parser = argparse.ArgumentParser(description="Transformer model training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=32, help="Maximum sequence length")
    parser.add_argument("--subset_fraction", type=float, default=0.1, help="Fraction of training data to use")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="Number of decoder layers")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size")
    args = parser.parse_args()

    device = torch.device(get_device())
    print(f'running on: {get_device()}')
    
    train_dataset = WMTDataset("train", args.max_length, subset_fraction=args.subset_fraction, vocab_size=args.vocab_size)
    val_dataset = WMTDataset("validation", args.max_length, vocab_size=args.vocab_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print('vocab size')
    print(args.vocab_size, train_dataset.tokenizer.vocab_size)
    
    model = Transformer(
        src_vocab_size=train_dataset.tokenizer.vocab_size,
        tgt_vocab_size=train_dataset.tokenizer.vocab_size,
        d_model=8,
        nhead=2,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=64,
        max_seq_length=args.max_length
    ).to(device)

    print(f"Vocabulary size: {min(args.vocab_size, train_dataset.tokenizer.vocab_size)}")
    print(f"Number of encoder layers: {args.num_encoder_layers}")
    print(f"Number of decoder layers: {args.num_decoder_layers}")
    print(f"The model has {count_parameters(model):,} trainable parameters")
    log_model_summary(model, input_size=(args.batch_size, args.max_length))
    #visualize_model(model, input_size=(args.batch_size, args.max_length))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    timer = Timer()
    for epoch in range(args.num_epochs):
        print(f"epoch {epoch+1}/{args.num_epochs}")

        timer.start()
        train_loss = train_model(model, train_loader, optimizer, device, accumulation_steps=4)
        epoch_time = timer.stop()

        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")
    
    print("training completed")

if __name__ == "__main__":
    main()