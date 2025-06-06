# src/eval.py

import torch
from sklearn.metrics import classification_report

def evaluate(model, dataloader, device, class_names):
    """
    Runs the model on the dataloader and prints a classification report.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs  = inputs.to(device)
            labels  = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))
