import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        lengths = batch['length'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        logit = model(input_ids, lengths)
        loss = criterion(logit, labels)
        loss.backward()

        clip_grad_norm_(parameters=model.parametrs(), max_norm=1.0)

        optimizer.step()

        total_loss += loss

        preds = torch.argmax(logit, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, precision, recall, f1


