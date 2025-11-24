import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_ch=1, code_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_ch, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc_enc = None
        self.code_dim = code_dim

    def forward(self, x):
        B = x.shape[0]
        feat = self.conv(x)
        if self.fc_enc is None:
            flat_dim = feat.numel() // B
            self.fc_enc = nn.Linear(flat_dim, self.code_dim).to(x.device)
            print(f"✅ Initialized fc_enc with input size {flat_dim}")
        flat = feat.reshape(B, -1)
        z = self.fc_enc(flat)
        return z
