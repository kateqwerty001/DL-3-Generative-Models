import torch
import torch.nn as nn


# ============================================================
# Weight Initialization
# ============================================================

def weights_init(module):

    classname = module.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


# ============================================================
# Generator Block
# ============================================================

class GeneratorBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(True)
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# Critic Block
# ============================================================

class CriticBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.InstanceNorm2d(out_channels, affine=True),

            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# Generator
# ============================================================

class Generator(nn.Module):

    """
    Input:
        z -> (B, latent_dim)

    Output:
        fake image -> (B, 3, 128, 128)
    """

    def __init__(
        self,
        latent_dim=128,
        feature_maps=64,
        out_channels=3
    ):

        super().__init__()

        self.net = nn.Sequential(

            # (B, latent_dim, 1, 1)
            nn.ConvTranspose2d(
                latent_dim,
                feature_maps * 16,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(feature_maps * 16),
            nn.ReLU(True),

            # 4x4
            GeneratorBlock(feature_maps * 16, feature_maps * 8),

            # 8x8
            GeneratorBlock(feature_maps * 8, feature_maps * 4),

            # 16x16
            GeneratorBlock(feature_maps * 4, feature_maps * 2),

            # 32x32
            GeneratorBlock(feature_maps * 2, feature_maps),

            # 64x64
            nn.ConvTranspose2d(
                feature_maps,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # 128x128
            nn.Tanh()
        )

    def forward(self, z):

        z = z.view(z.size(0), z.size(1), 1, 1)

        return self.net(z)


# ============================================================
# Critic
# ============================================================

class Critic(nn.Module):

    """
    Input:
        image -> (B, 3, 128, 128)

    Output:
        realism score -> (B,)
    """

    def __init__(
        self,
        in_channels=3,
        feature_maps=64
    ):

        super().__init__()

        self.net = nn.Sequential(

            # 128 -> 64
            nn.Conv2d(
                in_channels,
                feature_maps,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 32
            CriticBlock(feature_maps, feature_maps * 2),

            # 32 -> 16
            CriticBlock(feature_maps * 2, feature_maps * 4),

            # 16 -> 8
            CriticBlock(feature_maps * 4, feature_maps * 8),

            # 8 -> 4
            CriticBlock(feature_maps * 8, feature_maps * 16),

            nn.Conv2d(
                feature_maps * 16,
                1,
                kernel_size=4,
                stride=1,
                padding=0
            )
        )

    def forward(self, x):

        out = self.net(x)

        return out.view(-1)