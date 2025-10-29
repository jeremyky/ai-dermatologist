import os, torch, torch.nn as nn, numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from derm_data import make_loaders
from model_tiny import TinyDermNet

def evaluate_metrics(y_true, y_pred, class_names, epoch=None):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n---Validation Metrics---")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro F1   : {f1_macro:.4f}   (leaderboard-aligned)")
    print(f"Micro F1   : {f1_micro:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    print("\nPer-class F1:")
    for i, cls in enumerate(class_names):
        print(f"  {cls:<45} {per_class_f1[i]:.4f}")

    # lightweight logfile
    os.makedirs("../runs", exist_ok=True)
    with open("../runs/metrics_log.txt", "a") as f:
        f.write(f"epoch={epoch if epoch is not None else -1}, "
                f"acc={acc:.4f}, macroF1={f1_macro:.4f}, microF1={f1_micro:.4f}, "
                f"prec={prec:.4f}, rec={rec:.4f}\n")

    return f1_macro  # handy for model selection

def train_epoch(model, crit, opt, loader, device):
    model.train()
    losses = []; preds=[]; gts=[]
    for x,y in tqdm(loader, leave=False):
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward(); opt.step()
        losses.append(loss.item())
        preds.append(out.argmax(1).detach().cpu().numpy())
        gts.append(y.cpu().numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    return float(np.mean(losses)), f1_score(gts, preds, average='macro')

@torch.no_grad()
def eval_epoch(model, crit, loader, device):
    model.eval()
    losses = []; preds=[]; gts=[]
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = crit(out, y)
        losses.append(loss.item())
        preds.append(out.argmax(1).cpu().numpy())
        gts.append(y.cpu().numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    return float(np.mean(losses)), gts, preds

def main():
    data_dir = "../data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_w, classes = make_loaders(
        data_dir, batch_size=128, val_ratio=0.1, seed=1337
    )
    model = TinyDermNet(num_classes=len(classes)).to(device)
    crit = nn.CrossEntropyLoss(weight=class_w.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_f1, best_path = -1.0, "../models/best_scripted.pt"
    os.makedirs("../models", exist_ok=True)

    for epoch in range(12):
        tr_loss, tr_f1 = train_epoch(model, crit, opt, train_loader, device)
        va_loss, y_true, y_pred = eval_epoch(model, crit, val_loader, device)
        print(f"\nEpoch {epoch+1:02d} | train loss {tr_loss:.4f} f1 {tr_f1:.4f} | val loss {va_loss:.4f}")

        val_macro_f1 = evaluate_metrics(y_true, y_pred, classes, epoch=epoch+1)

        # save best TorchScript by macro-F1
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            scripted = torch.jit.script(model.eval())
            scripted.save(best_path)
            sz_mb = os.path.getsize(best_path)/1024/1024
            print(f"Saved best → {best_path} | size {sz_mb:.2f} MB | macro-F1 {best_f1:.4f}")

if __name__ == "__main__":
    main()
