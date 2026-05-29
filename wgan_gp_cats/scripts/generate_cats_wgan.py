import itertools
import sys

import torch
from torch.optim import Adam

sys.path.append("datasets")

from datasets.cat_datasets import get_cat_dataloaders

sys.path.append("src")

from wgan_gp_cats.src.models.wgan_gp import Generator, Critic, weights_init
from wgan_gp_cats.src.trainers.wgan_gp_trainer import WGAN_GP_Trainer


# ============================================================
# Hyperparameter grid (like your VQ-VAE grid search)
# ============================================================

LAMBDA_GPS = [5, 10, 15]
N_CRITICS = [2, 3, 5]
# LATENT_DIMS = [64, 128]
LATENT_DIMS = [128]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(lambda_gp, n_critic, latent_dim):

    print(f"\nRunning: λ={lambda_gp}, n_critic={n_critic}, z={latent_dim}")

    train_loader, _, _ = get_cat_dataloaders(
        root_dir="data",
        batch_size=64,
        image_size=128,
        model_type="gan"
    )

    G = Generator(latent_dim=latent_dim).to(DEVICE)
    C = Critic().to(DEVICE)

    G.apply(weights_init)
    C.apply(weights_init)

    g_opt = Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    c_opt = Adam(C.parameters(), lr=1e-4, betas=(0.0, 0.9))

    trainer = WGAN_GP_Trainer(
        generator=G,
        critic=C,
        g_optimizer=g_opt,
        c_optimizer=c_opt,
        device=DEVICE,
        latent_dim=latent_dim,
        lambda_gp=lambda_gp,
        n_critic=n_critic,
        run_name=f"wgan_gp_l{lambda_gp}_n{n_critic}_z{latent_dim}"
    )

    trainer.train(train_loader, num_epochs=50)


# ============================================================
# Grid search
# ============================================================

for lg, nc, ld in itertools.product(LAMBDA_GPS, N_CRITICS, LATENT_DIMS):
    run_experiment(lg, nc, ld)