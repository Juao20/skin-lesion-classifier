"""Boucle d'entraînement."""
import argparse
import os

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

# Phase 1 : bouchon. Plus tard -> from src.data import build_dataloaders
from src.data_stub import build_dataloaders
from src.model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    labels_idx = list(range(len(class_names)))
    recall_per_class = recall_score(
        y_true, y_pred, labels=labels_idx, average=None, zero_division=0
    )
    mel_idx = class_names.index("mel")

    return {
        "loss": running_loss / len(loader.dataset),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=labels_idx, average="macro", zero_division=0
        ),
        "recall_mel": recall_per_class[mel_idx],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, class_info = build_dataloaders(cfg)

    model = build_model(
        cfg["model"]["name"],
        cfg["model"]["num_classes"],
        cfg["model"]["pretrained"],
    ).to(device)

    class_weights = class_info["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    class_names = class_info["class_names"]
    output_dir = cfg["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, "best_model.pt")
    best_f1 = -1.0

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, criterion, device, class_names)
        print(
            f"Epoch {epoch + 1:2d}/{cfg['train']['epochs']} | "
            f"train_loss {train_loss:.4f} | val_loss {metrics['loss']:.4f} | "
            f"bal_acc {metrics['balanced_acc']:.4f} | "
            f"macro_f1 {metrics['macro_f1']:.4f} | "
            f"recall_mel {metrics['recall_mel']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "class_names": class_names,
                    "model_name": cfg["model"]["name"],
                    "num_classes": cfg["model"]["num_classes"],
                },
                best_path,
            )
            print(f"  -> nouveau meilleur macro_f1 {best_f1:.4f}, sauvé dans {best_path}")

    print(f"Entraînement terminé. Meilleur macro_f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
