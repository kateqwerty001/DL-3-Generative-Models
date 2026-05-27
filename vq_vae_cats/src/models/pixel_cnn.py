import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """
    Convolution with autoregressive mask.
    Type A: excludes current position (used in first layer)
    Type B: includes current position (used in subsequent layers)
    """
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")

        mask = torch.ones_like(self.weight)
        _, _, H, W = self.weight.shape
        mask[:, :, H // 2, W // 2 + (1 if mask_type == "B" else 0):] = 0
        mask[:, :, H // 2 + 1:] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ResidualBlockPixel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d("B", channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            MaskedConv2d("B", channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class PixelCNN(nn.Module):
    """
    Autoregressive prior for VQ-VAE.
    Predicts next codebook index given all previous ones.

    Input:  indices (B, H, W)         — integer codebook indices
    Output: logits  (B, K, H, W)      — distribution over K codes per position
    """
    def __init__(self, num_embeddings=512, hidden_dim=256, n_layers=8):
        super().__init__()
        self.K = num_embeddings

        self.input_embedding = nn.Embedding(num_embeddings, hidden_dim)

        # first layer: mask type A (excludes current position)
        self.first_conv = MaskedConv2d(
            "A", hidden_dim, hidden_dim, kernel_size=7, padding=3
        )

        # residual layers: mask type B (includes current position)
        self.residual_layers = nn.Sequential(
            *[ResidualBlockPixel(hidden_dim) for _ in range(n_layers)]
        )

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_embeddings, kernel_size=1),
        )

    def forward(self, indices):
        x = self.input_embedding(indices).permute(0, 3, 1, 2)
        x = self.first_conv(x)
        x = self.residual_layers(x)
        logits = self.output(x)
        return logits

    @torch.no_grad()
    def generate(self, shape, device, temperature=1.0):
        """
        Autoregressively sample a new grid of indices.
        shape: (B, H, W)
        """
        B, H, W = shape
        indices = torch.zeros(B, H, W, dtype=torch.long, device=device)

        self.eval()
        for i in range(H):
            for j in range(W):
                logits = self.forward(indices)
                logits_ij = logits[:, :, i, j]
                probs = F.softmax(logits_ij / temperature, dim=-1)
                sampled = torch.multinomial(probs, 1).squeeze(1)
                indices[:, i, j] = sampled

        return indices