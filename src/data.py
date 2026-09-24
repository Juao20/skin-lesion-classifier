"""Dataset, transforms et split train/val/test."""
import os
import glob
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def get_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def resolve_data_root(configured_root: str) -> str:
    kaggle_root = "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000"
    if os.path.isdir(configured_root):
        return configured_root
    if os.path.isdir(kaggle_root):
        return kaggle_root
    raise FileNotFoundError(
        f"Dataset introuvable. Essayé : '{configured_root}' et '{kaggle_root}'"
    )


def build_image_path_lookup(root_dir: str) -> dict:
    paths = glob.glob(os.path.join(root_dir, "HAM10000_images_part_*", "*.jpg"))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, image_paths: dict, class_names: list, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_paths = image_paths
        self.transform = transform
        # dict pour convertir "mel" -> 4, "nv" -> 5, etc. (ordre alphabétique)
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_paths[row["image_id"]]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row["dx"]]
        return image, label


def build_dataloaders(config: dict):
    data_cfg = config["data"]
    root = resolve_data_root(data_cfg["root"])

    df = pd.read_csv(os.path.join(root, "HAM10000_metadata.csv"))
    image_paths = build_image_path_lookup(root)

    # Split stratifié groupé par lésion (évite la fuite entre train/val)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config["seed"])
    train_idx, val_idx = next(sgkf.split(df, df["dx"], groups=df["lesion_id"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    train_transform = get_transforms(data_cfg["image_size"], train=True)
    val_transform = get_transforms(data_cfg["image_size"], train=False)

    train_ds = HAM10000Dataset(train_df, image_paths, CLASS_NAMES, train_transform)
    val_ds = HAM10000Dataset(val_df, image_paths, CLASS_NAMES, val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    # Poids de classe : inversement proportionnels à la fréquence
    counts = train_df["dx"].value_counts().reindex(CLASS_NAMES)
    class_weights = torch.tensor((1.0 / counts).values, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(CLASS_NAMES)

    class_info = {
        "class_names": CLASS_NAMES,
        "class_weights": class_weights,
    }

    return train_loader, val_loader, class_info