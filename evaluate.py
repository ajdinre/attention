import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    all_preds = []
    all_labels = []

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

            preds = torch.argmax(outputs, dim=-1)
            
            # exclude padding tokens (assuming 0 is the padding token)
            mask = output_tgt != 0
            preds = preds[mask]
            labels = output_tgt[mask]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    
    # calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1