import torch
import torch.nn as nn
from tqdm import tqdm


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src = batch["source_ids"].to(device)
            tgt = batch["target_ids"].to(device)

            input_tgt = tgt[:, :-1]
            output_tgt = tgt[:, 1:]

            outputs = model(src, input_tgt)

            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            output_tgt = output_tgt.contiguous().view(-1)

            loss = criterion(outputs, output_tgt)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
