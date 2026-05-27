"""
Train GPT prior on top of the best VQ-VAE checkpoint.
Phase 1 only: up to 200 epochs, lr=1e-4, patience=10
"""

import sys
import shutil
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.append("src")

from models.vq_vae import VQ_VAE
from trainers.pixel_cnn_trainer import collect_indices, IndexDataset
from models.gpt_prior import GPTPrior
from trainers.gpt_prior_trainer import GPTPriorTrainer
from cat_datasets import get_cat_dataloaders

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────

# best VQ-VAE from grid search — adjust beta / K to whichever was best
BETA           = 1.5
K              = 512
VQVAE_CKPT     = f"vqvae_models/checkpoints_{BETA}_{K}/vqvae_best.pt"

INDICES_DIR    = Path(f"gpt_models/gpt_indices_{BETA}_{K}")
DATA_DIR       = "data"
IMAGE_SIZE     = 128
BATCH_SIZE     = 64
LATENT_H       = 32          # 128 / 4
LATENT_W       = 32

# GPT architecture
HIDDEN_DIM  = 256
N_LAYERS    = 8
N_HEADS     = 8
DROPOUT     = 0.1

# Training
LR          = 1e-3
MAX_EPOCHS  = 500
PATIENCE    = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Destination path for the final model
GPT_SAVE_DIR = Path(f"gpt_models/checkpoints_{BETA}_{K}")
GPT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────

def load_vqvae():
    ckpt  = torch.load(VQVAE_CKPT, map_location=DEVICE)
    model = VQ_VAE(in_channels=3, hidden_dim=128, embedding_dim=256,
                   num_embedding=K, n_residual=2, beta=BETA).to(DEVICE)
    
    # Handle dict with "model" key or bare state_dict
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"VQ-VAE loaded (beta={BETA}, K={K})")
    return model


def get_index_loaders(vqvae):
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    train_idx = INDICES_DIR / "train_indices.pt"
    val_idx   = INDICES_DIR / "val_indices.pt"

    train_img, val_img, _ = get_cat_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
        num_workers=1, model_type="vqvae",
    )

    if not train_idx.exists():
        print("Collecting train indices...")
        collect_indices(vqvae, train_img, DEVICE, train_idx)
    if not val_idx.exists():
        print("Collecting val indices...")
        collect_indices(vqvae, val_img, DEVICE, val_idx)

    train_loader = DataLoader(IndexDataset(train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(IndexDataset(val_idx),   batch_size=BATCH_SIZE,
                              shuffle=False)
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    return train_loader, val_loader


def build_model():
    return GPTPrior(
        num_embeddings=K,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)


def main():
    print(f"Device: {DEVICE}")

    vqvae                    = load_vqvae()
    train_loader, val_loader = get_index_loaders(vqvae)

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training GPT Prior: lr={LR}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}")
    print(f"{'='*60}")

    model      = build_model()
    optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    trainer    = GPTPriorTrainer(model, optimizer, DEVICE,
                                 run_name=f"gpt_{BETA}_{K}")

    best_val  = trainer.train(train_loader, val_loader,
                                max_epochs=MAX_EPOCHS, patience=PATIENCE)
    
    best_path = Path(f"gpt_models/checkpoint_{BETA}_{K}/gpt_best.pt")
    print(f"\nTraining done. Best val_ce={best_val:.4f}")

    # ── Pick overall best and move to desired directory ──────────────────────
    dest = GPT_SAVE_DIR / "gpt_best.pt"
    if best_path.exists():
        shutil.copy(best_path, dest)
        print(f"\n✅ Best GPT prior successfully saved to {dest}")
    else:
        print(f"\n⚠️ Warning: Could not find {best_path} to copy.")


if __name__ == "__main__":
    main()