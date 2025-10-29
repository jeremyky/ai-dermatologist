import os, random, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
import numpy as np

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif')

def make_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomResizedCrop(120, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(120),
        transforms.CenterCrop(120),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf


class SkinDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.items = []
        for c in self.classes:
            p = os.path.join(root, c)
            for f in os.listdir(p):
                if f.lower().endswith(VALID_EXTS):
                    self.items.append((os.path.join(p, f), self.class_to_idx[c]))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        try:
            x = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # fallback: blank image
            x = Image.new("RGB", (120,120), (0,0,0))
        if self.transform: x = self.transform(x)
        return x, y


def stratified_split(labels, val_ratio=0.1, seed=1337):
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    idx = np.arange(len(labels))
    val_idx = []
    for c in set(labels):
        c_idx = idx[labels==c]
        rng.shuffle(c_idx)
        k = max(1, int(len(c_idx)*val_ratio))
        val_idx.extend(c_idx[:k].tolist())
    val_idx = sorted(set(val_idx))
    train_idx = [i for i in idx.tolist() if i not in val_idx]
    return train_idx, val_idx


def class_weights_from_counts(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    w = 1.0 / counts
    w *= (n_classes / w.sum())
    return torch.tensor(w, dtype=torch.float)


def make_loaders(data_dir, batch_size=128, val_ratio=0.1, seed=1337, use_weighted_sampler=False):
    train_tf, val_tf = make_transforms()
    full = SkinDataset(data_dir, transform=None)
    labels = [y for _, y in full.items]

    split_file = os.path.join(data_dir, "_split_indices.pt")
    if os.path.exists(split_file):
        d = torch.load(split_file)
        train_idx, val_idx = d["train_idx"], d["val_idx"]
    else:
        train_idx, val_idx = stratified_split(labels, val_ratio=val_ratio, seed=seed)
        torch.save({"train_idx": train_idx, "val_idx": val_idx}, split_file)

    train_set = Subset(SkinDataset(data_dir, transform=train_tf), train_idx)
    val_set   = Subset(SkinDataset(data_dir, transform=val_tf), val_idx)
    cw = class_weights_from_counts([labels[i] for i in train_idx], n_classes=len(full.classes))

    if use_weighted_sampler:
        counts = np.bincount([labels[i] for i in train_idx], minlength=len(full.classes)).astype(float)
        sample_w = [1.0 / counts[labels[i]] for i in train_idx]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(train_idx), replacement=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, cw, full.classes


if __name__ == "__main__":
    train_loader, val_loader, cw, classes = make_loaders("../data", batch_size=8)
    print("Classes:", classes)
    print("Class weights:", cw)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels[:8]}")
