import torch
from torch.utils.data import DataLoader
from model.transformer import Transformer

from data.dataset import WMTDataset
from train import train_model
from evaluate import evaluate_model

from utils import get_device

def main():
    device = torch.device(get_device())
    
    # hp
    batch_size = 256
    num_epochs = 10
    learning_rate = 0.0001
    max_length = 128
    
    train_dataset = WMTDataset("train", max_length, subset_fraction=0.1)
    val_dataset = WMTDataset("validation", max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = Transformer(
        src_vocab_size=train_dataset.tokenizer.vocab_size,
        tgt_vocab_size=train_dataset.tokenizer.vocab_size,
        d_model=16,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        max_seq_length=max_length
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        print(f"epoch {epoch+1}/{num_epochs}")
        train_model(model, train_loader, optimizer, device, accumulation_steps=4)
        
        val_loss = evaluate_model(model, val_loader, device)
        print(f"validation Loss: {val_loss:.4f}")
    
    print("training completed")

if __name__ == "__main__":
    main()
