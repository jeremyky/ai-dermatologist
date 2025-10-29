import os, zipfile, requests, time, random
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# 0. Setup
# ======================================================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ======================================================
# 1. Download Dataset
# ======================================================
url = "http://hadi.cs.virginia.edu:8000/download/train-dataset"
zip_path = "train_dataset.zip"
extract_dir = "train_dataset"

if not os.path.exists(zip_path):
    print("Downloading dataset...")
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(zip_path, "wb") as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
        for chunk in resp.iter_content(1024):
            f.write(chunk)
            bar.update(len(chunk))
else:
    print("Dataset already exists")

if not os.path.exists(extract_dir):
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extracted")
else:
    print("Already extracted")

# ======================================================
# 2. Data utilities (formerly derm_data.py)
# ======================================================
class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif')
        self.image_paths, self.labels = [], []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(valid_exts):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path, label = self.image_paths[idx], self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Could not load {path}: {e}")
            img = Image.new("RGB", (120,120))
        if self.transform:
            img = self.transform(img)
        return img, label


def make_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomResizedCrop(120, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((120,120)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf


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


def make_loaders(data_dir, batch_size=128, val_ratio=0.1, use_weighted_sampler=False):
    train_tf, val_tf = make_transforms()
    full = SkinDataset(data_dir, transform=None)
    labels = np.array(full.labels)
    n_classes = len(full.classes)

    # Split once deterministically
    split_file = os.path.join(data_dir, "_split_indices.pt")
    if os.path.exists(split_file):
        d = torch.load(split_file)
        train_idx, val_idx = d["train_idx"], d["val_idx"]
    else:
        train_idx, val_idx = stratified_split(labels, val_ratio)
        torch.save({"train_idx": train_idx, "val_idx": val_idx}, split_file)

    train_ds = Subset(SkinDataset(data_dir, transform=train_tf), train_idx)
    val_ds   = Subset(SkinDataset(data_dir, transform=val_tf), val_idx)

    if use_weighted_sampler:
        counts = np.bincount(labels[train_idx], minlength=n_classes).astype(float)
        sample_w = [1.0 / counts[l] for l in labels[train_idx]]
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, n_classes, full.classes


# ======================================================
# 3. Model definition (from model_tiny.py)
# ======================================================
def dwpw(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )

class TinyDermNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        c = [16, 32, 64, 96, 128, 160]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            dwpw(c[0], c[1], 1),
            dwpw(c[1], c[2], 2),
            dwpw(c[2], c[2], 1),
            dwpw(c[2], c[3], 2),
            dwpw(c[3], c[3], 1),
            dwpw(c[3], c[4], 2),
            dwpw(c[4], c[5], 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

# ======================================================
# 4. Evaluation utilities (from train.py)
# ======================================================
def evaluate_metrics(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    print("\nEvaluation Metrics")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Macro F1  : {f1_macro:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Per-class F1
    per_f1 = f1_score(y_true, y_pred, average=None)
    for i, cls in enumerate(class_names):
        print(f"  {cls:<35}: F1 = {per_f1[i]:.4f}")
    return acc, f1_macro


# ======================================================
# 5. Training pipeline
# ======================================================
data_dir = "train_dataset/train_dataset"
train_loader, val_loader, num_classes, class_names = make_loaders(data_dir, batch_size=256)
model = TinyDermNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
epochs = 10

print(f"Training on {num_classes} classes ({len(train_loader.dataset)} train, {len(val_loader.dataset)} val)")

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds.append(torch.argmax(out, dim=1).cpu())
            labels_all.append(labels.cpu())
    preds = torch.cat(preds)
    labels_all = torch.cat(labels_all)
    acc, f1 = evaluate_metrics(labels_all, preds, class_names)
    print(f"Epoch {epoch+1}: TrainLoss={train_loss:.4f}, ValAcc={acc:.4f}, ValF1={f1:.4f}")

# ======================================================
# 6. Save model as TorchScript (<5MB)
# ======================================================
model.eval()
scripted = torch.jit.script(model)
save_path = "TinyDermNet_scripted.pt"
scripted.save(save_path)
size_mb = os.path.getsize(save_path)/1024/1024
print(f"Saved model {save_path} ({size_mb:.2f} MB)")

# ======================================================
# 7. (Optional) Submission utilities
# ======================================================
def submit_model(token: str, model_path: str, server_url="http://hadi.cs.virginia.edu:8000"):
    with open(model_path, "rb") as f:
        files = {"file": f}
        data = {"token": token}
        resp = requests.post(f"{server_url}/submit", data=data, files=files)
        try:
            msg = resp.json()
            print(msg)
        except Exception:
            print("Invalid server response:", resp.text)

def check_submission_status(token):
    url = f"http://hadi.cs.virginia.edu:8000/submission-status/{token}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return
    for a in resp.json():
        score = a.get("score", "N/A")
        print(f"Attempt {a['attempt']}: Score={score}, Size={a.get('model_size','?')}MB, Status={a['status']}")

# Example (comment out if not submitting automatically)
# my_token = "your_token_here"
# submit_model(my_token, save_path)
# check_submission_status(my_token)
