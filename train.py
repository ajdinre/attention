import torch
import torch.nn as nn
from tqdm import tqdm

# accumulation_steps is used to simulate larger batches
# speeds up the training time

def train_model(model, dataloader, optimizer, device, accumulation_steps=1):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is the padding index

    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        src = batch['source_ids'].to(device)
        tgt = batch['target_ids'].to(device)
        
        # shift the target sequence for next token prediction
        input_tgt = tgt[:, :-1]
        output_tgt = tgt[:, 1:]
        
        outputs = model(src, input_tgt)
        
        # reshape outputs and targets for loss calculation
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        output_tgt = output_tgt.contiguous().view(-1)
        
        loss = criterion(outputs, output_tgt)
        loss = loss / accumulation_steps  # normalize the loss
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss
