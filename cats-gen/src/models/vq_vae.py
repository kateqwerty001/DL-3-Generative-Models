import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block for VQ-VAE."""
    
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)
    

class Encoder(nn.Module):
    """
    Downsamples spatial dimensions by a factor of 4.
    Input: 128x128x3 -> Output: 32x32x256
    """
    def __init__(self, in_channels=3, hidden_dim=128, embedding_dim=256, n_residual=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            *[ResidualBlock(hidden_dim) for _ in range(n_residual)], 
            nn.Conv2d(hidden_dim, embedding_dim, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class VectorQuantizer(nn.Module):
    """
    Quantizes encoder output to nearest codebook vector.
    Input: (B, D, H, W) -> Output: z_q (B, D, H, W), indices (B, H, W), loss
    """
    def __init__(self, num_embeddings=512, embedding_dim=256, beta=0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.codebook = nn.Embedding(self.K, self.D)
        nn.init.uniform_(self.codebook.weight, -1/self.K, 1/self.K)

    def forward(self, z_encoder):
        # (B, D, H, W) -> (B, H, W, D)
        z_encoder = z_encoder.permute(0, 2, 3, 1).contiguous()
        B, H, W, D = z_encoder.shape

        # L2 distances between each encoder vector and each codebook entry
        flat = z_encoder.view(-1, D)
        distances = (
            flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.weight.t() + self.codebook.weight.pow(2).sum(1)
        )

        indices = distances.argmin(1)
        z_quantized = self.codebook(indices).view(B, H, W, D)

        # losses
        codebook_loss = F.mse_loss(z_quantized, z_encoder.detach())
        commitment_loss = F.mse_loss(z_encoder, z_quantized.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # straight-through estimator: forward=z_q, backward=z_e
        z_quantized = z_encoder + (z_quantized - z_encoder).detach()

        # (B, H, W, D) -> (B, D, H, W)
        z_quantized = z_quantized.permute(0, 3, 1, 2).contiguous()
        indices = indices.view(B, H, W)

        return z_quantized, indices, loss

    def codebook_utilization(self, indices):
        """Fraction of codebook entries used in this batch (diagnostic)."""
        unique = indices.unique().numel()
        ratio = unique/self.K
        return ratio


class Decoder(nn.Module):
    """
    Mirror of Encoder. Upsamples spatial dimensions by a factor of 4.
    Input: 32x32x256 -> Output: 128x128x3
    """
    def __init__(self, out_channels=3, hidden_dim=128, embedding_dim=256, n_residual=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            *[ResidualBlock(hidden_dim) for _ in range(n_residual)],
            # 32x32 -> 64x64
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), 
        )

    def forward(self, z_quantized):
        return self.net(z_quantized)

    
class VQ_VAE(nn.Module):
    """
    VQ-VAE model combining Encoder, VectorQuantizer, and Decoder.
    """
    def __init__(self, in_channels=3, hidden_dim=128, embedding_dim=256, num_embedding=512, n_residual=2, beta=0.25):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, embedding_dim, n_residual)
        self.quantizer = VectorQuantizer(num_embedding, embedding_dim, beta)
        self.decoder = Decoder(in_channels, hidden_dim, embedding_dim, n_residual)

    def forward(self, x):
        z_encoder = self.encoder(x)
        z_quantized, indices, vq_loss = self.quantizer(z_encoder)
        x_reconstructed = self.decoder(z_quantized)
        return x_reconstructed, vq_loss, indices

    def encode_indices(self, x):
        """Returns codebook indices only. """
        with torch.no_grad():
            z_encoder = self.encoder(x)
            _, indices, _ = self.quantizer(z_encoder)

        return indices

    def decode_indiced(self, indices):
        """Decodes codebook indices back to images """
        with torch.no_grad():
            z_quantized = self.quantizer.codebook(indices)
            z_quantized = z_quantized.permute(0, 3, 1, 2).contiguous()
            x_reconstructed =self.decoder(z_quantized)

        return x_reconstructed