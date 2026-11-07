"""Train a frozen EfficientNet-B0 backbone, select threshold on validation only."""

import argparse
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from glaucoma.data import FundusDataset, check_split_overlap
from glaucoma.metrics import binary_metrics, select_threshold
from glaucoma.pipeline import CLASSES, PREPROCESSING, build_model


def extract_features(model, dataset, device, batch_size):
    """A frozen backbone with deterministic preprocessing only needs one pass."""
    model.eval()
    features, targets = [], []
    loader = DataLoader(dataset, batch_size=batch_size)
    with torch.inference_mode():
        for index, (images, labels) in enumerate(loader):
            output = model.features(images.to(device))
            output = model.avgpool(output).flatten(1)
            features.append(output.cpu())
            targets.append(labels)
            if (index + 1) % 25 == 0 or index + 1 == len(loader):
                print(f"  Backbone features: {min((index + 1) * batch_size, len(dataset))}/{len(dataset)}", flush=True)
    # Clone outside inference mode so the classifier can save inputs for backward.
    return TensorDataset(torch.cat(features).clone(), torch.cat(targets).clone())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(2)
    train = FundusDataset(args.data / "train")
    val = FundusDataset(args.data / "val")
    check_split_overlap(train, val)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    print(f"Training on {device}; {len(train)} train / {len(val)} validation images.")
    print("ImageNet weights download on first training run. No dataset is downloaded.")
    model = build_model(pretrained=True).to(device)
    for p in model.features.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    print("Extracting frozen backbone features once for train and validation.", flush=True)
    train_features = extract_features(model, train, device, args.batch_size)
    val_features = extract_features(model, val, device, args.batch_size)
    loader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_features, batch_size=args.batch_size)
    best_loss = float("inf")
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        model.features.eval()  # Also freeze batchnorm statistics in the backbone.
        train_loss = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model.classifier(features), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        model.eval()
        val_loss, all_labels, probabilities = 0.0, [], []
        with torch.inference_mode():
            for features, labels in val_loader:
                logits = model.classifier(features.to(device))
                val_loss += criterion(logits, labels.to(device)).item() * len(labels)
                all_labels.extend(labels.tolist())
                probabilities.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
        val_loss /= len(val)
        print(f"Epoch {epoch + 1:02d}: train loss {train_loss / len(train):.4f}, val loss {val_loss:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss / len(train), "validation_loss": val_loss})
        args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
        if val_loss < best_loss:
            best_loss = val_loss
            threshold = select_threshold(all_labels, probabilities)
            torch.save({
                "architecture": "efficientnet_b0", "classes": CLASSES,
                "preprocessing": PREPROCESSING, "threshold": threshold,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch + 1, "seed": args.seed,
                "training_samples": len(train),
                "dataset_provenance": json.loads((args.data / "provenance.json").read_text())
                if (args.data / "provenance.json").exists() else {"source": "user_supplied"},
                "validation_metrics": binary_metrics(all_labels, probabilities, threshold),
                "validation_loss": val_loss,
                "status": "research_only_unvalidated",
            }, args.output)
    print(f"Saved {args.output}. Run evaluate.py on the held-out test split.")


if __name__ == "__main__":
    main()
