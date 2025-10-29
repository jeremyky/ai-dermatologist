'''
Freeze a split (no leakage).
Save indices once so every run compares apples-to-apples.

Set up loaders + transforms.
Include the augmentations we discussed in the EDA (scale/zoom, light color jitter, occasional blur).

Handle imbalance.
Compute class_weights for CrossEntropyLoss (or use a WeightedRandomSampler).

Train a tiny CNN (<5 MB).
Start with a depthwise-separable MobileNet-style net (~0.8–1.1M params).

Log metrics that match grading.
Track macro-F1 (and per-class F1) on the held-out split.

Export TorchScript + size-check.
Ensure input is [B,3,120,120], file <5 MB. 
'''
import torch, torch.nn as nn

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
        c = [16, 32, 64, 96, 128, 160]  # keep channels modest
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
        self.head = nn.Linear(c[5], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.head(x)

if __name__ == "__main__":
    m = TinyDermNet(10)
    x = torch.randn(2,3,120,120)
    y = m(x)
    scripted = torch.jit.script(m.eval())
    scripted.save("TinyDermNet.pt")
    import os
    print("Params:", sum(p.numel() for p in m.parameters()))
    print("File MB:", os.path.getsize("TinyDermNet.pt")/1024/1024)
