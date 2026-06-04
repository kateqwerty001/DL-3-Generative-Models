import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
VQVAE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = VQVAE_ROOT.parent
SRC_DIR = VQVAE_ROOT / "src"
DATASETS_DIR = REPO_ROOT / "datasets"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DATASETS_DIR))

from models.vq_vae import VQ_VAE
from models.gpt_prior import GPTPrior
from trainers.pixel_cnn_trainer import collect_indices
from trainers.gpt_prior_trainer import GPTPriorTrainer, IndexDataset
from cat_datasets import get_cat_dataloaders

BETA = 1.5
K = 512
IMAGE_SIZE = 128
BATCH_SIZE = 64
LATENT_H = 32
LATENT_W = 32
HIDDEN_DIM = 256
N_LAYERS = 8
N_HEADS = 8
DROPOUT = 0.1
LR = 1e-3
MAX_EPOCHS = 500
PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--k", type=int, default=K)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder with cat .jpg files (default: ~/data/cats)",
    )
    return p.parse_args()


def resolve_data_dir(data_dir):
    if data_dir is not None:
        data_dir = Path(data_dir).expanduser().resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        return data_dir

    candidates = [
        Path.home() / "data" / "cats",
        REPO_ROOT / "data" / "cats",
        VQVAE_ROOT / "data" / "cats",
        Path.home() / "data",
        REPO_ROOT / "data",
        VQVAE_ROOT / "data",
    ]
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_dir() and any(candidate.rglob("*.jpg")):
            return candidate
    raise FileNotFoundError(
        "Could not find cat images. Pass --data-dir (e.g. ~/data/cats). "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def paths(beta, k, data_dir):
    vqvae_ckpt = (
        VQVAE_ROOT / "models" / "vq_vae_models" / f"checkpoints_{beta}_{k}" / "vqvae_best.pt"
    )
    indices_dir = VQVAE_ROOT / "gpt_indices" / f"gpt_indices_{beta}_{k}"
    data_dir = resolve_data_dir(data_dir)
    save_root = VQVAE_ROOT / "models" / "gpt_prior_models" / "gpt_models"
    log_dir = VQVAE_ROOT / "training_logs" / "gpt_prior"
    return vqvae_ckpt, indices_dir, data_dir, save_root, log_dir


def load_vqvae(ckpt_path, beta, k):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = VQ_VAE(
        in_channels=3,
        hidden_dim=128,
        embedding_dim=256,
        num_embedding=k,
        n_residual=2,
        beta=beta,
    ).to(DEVICE)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"VQ-VAE loaded (beta={beta}, K={k}) from {ckpt_path}")
    return model


def get_index_loaders(vqvae, indices_dir, data_dir):
    indices_dir.mkdir(parents=True, exist_ok=True)
    train_idx = indices_dir / "train_indices.pt"
    val_idx = indices_dir / "val_indices.pt"

    train_img, val_img, _ = get_cat_dataloaders(
        str(data_dir),
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=1,
        model_type="vqvae",
    )

    if not train_idx.exists():
        print("Collecting train indices...")
        collect_indices(vqvae, train_img, DEVICE, train_idx)
    if not val_idx.exists():
        print("Collecting val indices...")
        collect_indices(vqvae, val_img, DEVICE, val_idx)

    train_loader = DataLoader(
        IndexDataset(train_idx), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(IndexDataset(val_idx), batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    return train_loader, val_loader


def build_model(k):
    return GPTPrior(
        num_embeddings=k,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)


def main():
    args = parse_args()
    beta, k = args.beta, args.k
    vqvae_ckpt, indices_dir, data_dir, save_root, log_dir = paths(beta, k, args.data_dir)

    if not vqvae_ckpt.exists():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {vqvae_ckpt}")

    print(f"Device: {DEVICE}")
    print(f"Data: {data_dir}")

    vqvae = load_vqvae(vqvae_ckpt, beta, k)
    train_loader, val_loader = get_index_loaders(vqvae, indices_dir, data_dir)

    print(f"\n{'=' * 60}")
    print(f"Training GPT Prior: beta={beta} K={k} lr={LR} max_epochs={MAX_EPOCHS} patience={PATIENCE}")
    print(f"{'=' * 60}")

    model = build_model(k)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    trainer = GPTPriorTrainer(
        model,
        optimizer,
        DEVICE,
        run_name=f"gpt_{beta}_{k}",
        save_dir=str(save_root),
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    trainer.log_dir = log_dir
    trainer.log_path = log_dir / f"gpt_{beta}_{k}_train_log.csv"

    best_val = trainer.train(train_loader, val_loader, max_epochs=MAX_EPOCHS, patience=PATIENCE)
    best_path = save_root / f"gpt_{beta}_{k}" / "best.pt"
    print(f"\nTraining done. Best val_ce={best_val:.4f}")
    print(f"Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
