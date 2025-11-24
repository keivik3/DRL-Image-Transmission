import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, code_dim=128, n_prev=5, output_ch=1):
        super().__init__()
        self.n_prev = n_prev
        self.code_dim = code_dim
        total_dim = (n_prev + 1) * code_dim

        self.fc_dec = nn.Linear(total_dim, 64 * 8 * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, output_ch, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def decode(self, z, prev_latents=None):
        if prev_latents is not None:
            # prev_latents: (1, n_prev, code_dim)
            B = z.size(0)
            prev = prev_latents.view(B, self.n_prev * self.code_dim)
            z_cat = torch.cat([z, prev], dim=1)
        else:
            B = z.size(0)
            zeros = torch.zeros(B, self.n_prev * self.code_dim, device=z.device)
            z_cat = torch.cat([z, zeros], dim=1)

        feat = self.fc_dec(z_cat)
        feat = feat.view(-1, 64, 8, 8)
        rec = self.deconv(feat)
        return rec

    # Make forward = decode for convenience
    def forward(self, z, prev_latents=None):
        return self.decode(z, prev_latents)
