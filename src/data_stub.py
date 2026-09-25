"""Bouchon temporaire pour build_dataloaders.

À SUPPRIMER quand src/data.py expose la vraie build_dataloaders.
Sert uniquement à tester que la boucle d'entraînement s'emboîte.
"""
import torch
from torch.utils.data import TensorDataset, DataLoader


def build_dataloaders(config):
    n_classes = config["model"]["num_classes"]
    fake_images = torch.randn(64, 3, 224, 224)
    fake_labels = torch.randint(0, n_classes, (64,))
    ds = TensorDataset(fake_images, fake_labels)
    loader = DataLoader(ds, batch_size=config["train"]["batch_size"])
    class_info = {
        "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "class_weights": torch.ones(n_classes),
    }
    return loader, loader, class_info
