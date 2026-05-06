from sklearn.metrics import accuracy_score
import torch
from torch import nn

def evaluate(model, loader, device, criterion=nn.CrossEntropyLoss(reduction='sum')):
    model.eval()
    total_loss, total_len, all_preds, all_labels, all_probs, all_logits = 0.0, 0, [], [], [], []
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_len += len(labels)
            preds = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)[torch.arange(preds.size(0)), preds]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / total_len, acc, all_labels, all_preds, all_probs, all_logits
 
