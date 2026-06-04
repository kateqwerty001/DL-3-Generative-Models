"""
ResNet50 species scores for all 9 VQ-VAE + GPT-Prior configs.

Generates N_GEN images in memory (optional save to disk), scores each with
ResNet50 ImageNet softmax: cat breeds 281-285, dog breeds 151-268.

Run from vq_vae_cats/scripts/:

    python evaluate_vqvae_gpt.py --temperature 1.0
"""

import argparse
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from torchvision.models import ResNet50_Weights, resnet50

SCRIPT_DIR = Path(__file__).resolve().parent
VQVAE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = VQVAE_ROOT.parent
SRC_DIR = VQVAE_ROOT / "src"

VQVAE_SEARCH_DIRS = [
    VQVAE_ROOT / "models" / "vq_vae_models",
    VQVAE_ROOT / "models" / "vqvae_models",
]
GPT_SEARCH_DIRS = [
    VQVAE_ROOT / "models" / "gpt_prior_models" / "gpt_models",
    VQVAE_ROOT / "models" / "gpt_models",
]

sys.path.insert(0, str(SRC_DIR))

from models.gpt_prior import GPTPrior
from models.vq_vae import VQ_VAE

BETAS = [0.5, 1.0, 1.5]
CODEBOOK_SIZES = [128, 256, 512]

LATENT_H = 32
LATENT_W = 32
HIDDEN_DIM = 256
N_LAYERS = 8
N_HEADS = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CAT_CLASSES = list(range(281, 286))
DOG_CLASSES = list(range(151, 269))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="VQ-VAE+GPT ResNet50 species evaluation")
    p.add_argument("--n-gen", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Results CSV (default: outputs/gpt_species_score_results_tau{T}.csv)",
    )
    p.add_argument(
        "--save-images",
        type=Path,
        default=None,
        help="If set, save generated PNGs under save-images/beta{B}_K{K}/",
    )
    p.add_argument("--beta", type=float, default=None, help="Run single config only")
    p.add_argument("--k", type=int, default=None, help="Run single config only")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_vqvae_ckpt(beta: float, k: int):
    for base in VQVAE_SEARCH_DIRS:
        path = base / f"checkpoints_{beta}_{k}" / "vqvae_best.pt"
        if path.exists():
            return path
    return None


def find_gpt_ckpt(beta: float, k: int):
    for base in GPT_SEARCH_DIRS:
        path = base / f"gpt_{beta}_{k}" / "best.pt"
        if path.exists():
            return path
    return None


def load_vqvae(beta: float, k: int, ckpt_path: Path) -> VQ_VAE:
    model = VQ_VAE(
        in_channels=3,
        hidden_dim=128,
        embedding_dim=256,
        num_embedding=k,
        n_residual=2,
        beta=beta,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def load_gpt(k: int, ckpt_path: Path) -> GPTPrior:
    model = GPTPrior(
        num_embeddings=k,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def load_scorer():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    model.to(DEVICE)
    return model


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_v = statistics.mean(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_v, std_v


@torch.no_grad()
def species_scores_batch(
    scorer,
    images: torch.Tensor,
) -> tuple[list[float], list[float]]:
    x = TF.normalize(images, IMAGENET_MEAN, IMAGENET_STD)
    probs = torch.softmax(scorer(x), dim=-1)
    cat_scores = probs[:, CAT_CLASSES].sum(dim=1).cpu().tolist()
    dog_scores = probs[:, DOG_CLASSES].sum(dim=1).cpu().tolist()
    return cat_scores, dog_scores


@torch.no_grad()
def generate_and_score(
    vqvae: VQ_VAE,
    gpt: GPTPrior,
    scorer,
    n_gen: int,
    batch_size: int,
    temperature: float,
    seed: int,
    save_dir,
) -> tuple[float, float, float, float]:
    cat_scores: list[float] = []
    dog_scores: list[float] = []
    generated = 0
    batch_idx = 0

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    while generated < n_gen:
        set_seed(seed + batch_idx)

        n_this = min(batch_size, n_gen - generated)
        indices = gpt.generate((n_this, LATENT_H, LATENT_W), DEVICE, temperature)
        images = vqvae.decode_indices(indices).clamp(0, 1)

        batch_cat, batch_dog = species_scores_batch(scorer, images)
        cat_scores.extend(batch_cat)
        dog_scores.extend(batch_dog)

        if save_dir is not None:
            for i, img in enumerate(images.cpu()):
                vutils.save_image(img, save_dir / f"{generated + i:05d}.png")

        generated += n_this
        batch_idx += 1
        print(f"  {generated}/{n_gen}", end="\r")

    print()
    return (*mean_std(cat_scores), *mean_std(dog_scores))


def iter_configs(beta_filter, k_filter):
    for beta in BETAS:
        for k in CODEBOOK_SIZES:
            if beta_filter is not None and beta != beta_filter:
                continue
            if k_filter is not None and k != k_filter:
                continue
            yield beta, k


def main():
    args = parse_args()

    out_csv = args.out_csv
    if out_csv is None:
        tag = str(args.temperature).replace(".", "p")
        out_csv = VQVAE_ROOT / "outputs" / f"gpt_species_score_results_tau{tag}.csv"
    out_csv = out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"N_GEN={args.n_gen}  temperature={args.temperature}  seed={args.seed}")
    print(f"Results -> {out_csv}\n")

    scorer = load_scorer()
    rows: list[tuple] = []

    for beta, k in iter_configs(args.beta, args.k):
        print("=" * 60)
        print(f"beta={beta}  K={k}")
        print("=" * 60)

        vqvae_path = find_vqvae_ckpt(beta, k)
        gpt_path = find_gpt_ckpt(beta, k)
        if vqvae_path is None or gpt_path is None:
            print("  SKIP missing checkpoint")
            if vqvae_path is None:
                print(f"    VQ-VAE not found for beta={beta} K={k}")
            if gpt_path is None:
                print(f"    GPT not found for beta={beta} K={k}")
            continue

        save_dir = None
        if args.save_images is not None:
            save_dir = (
                args.save_images.resolve()
                / f"beta{beta}_K{k}_tau{str(args.temperature).replace('.', 'p')}"
            )

        vqvae = load_vqvae(beta, k, vqvae_path)
        gpt = load_gpt(k, gpt_path)
        print(f"  VQ-VAE: {vqvae_path}")
        print(f"  GPT:    {gpt_path}")
        print(f"  Generating and scoring {args.n_gen} images...")

        cat_mean, cat_std, dog_mean, dog_std = generate_and_score(
            vqvae,
            gpt,
            scorer,
            args.n_gen,
            args.batch_size,
            args.temperature,
            args.seed,
            save_dir,
        )

        print(
            f"  cat: {cat_mean:.4f} +/- {cat_std:.4f}  |  "
            f"dog: {dog_mean:.4f} +/- {dog_std:.4f}"
        )

        rows.append((beta, k, cat_mean, cat_std, dog_mean, dog_std))

        del vqvae, gpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(out_csv, "w") as f:
        f.write(
            "beta,K,temperature,n_gen,cat_score,cat_score_std,dog_score,dog_score_std\n"
        )
        for beta, k, cat_m, cat_s, dog_m, dog_s in rows:
            f.write(
                f"{beta},{k},{args.temperature},{args.n_gen},"
                f"{cat_m:.6f},{cat_s:.6f},{dog_m:.6f},{dog_s:.6f}\n"
            )

    print("\n" + "=" * 60)
    print("FINAL (sorted by cat_score)")
    print("=" * 60)
    for beta, k, cat_m, cat_s, dog_m, dog_s in sorted(rows, key=lambda r: r[2], reverse=True):
        print(
            f"  beta={beta}  K={k:3d}  cat={cat_m:.4f}+/-{cat_s:.4f}  "
            f"dog={dog_m:.4f}+/-{dog_s:.4f}"
        )
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
