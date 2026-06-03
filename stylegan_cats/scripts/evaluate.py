"""
ResNet50 species-score evaluation for StyleGAN2-ADA cat models.

For each network-snapshot-*.pkl (training timestamp / kimg):
  - generate 500 images in memory (not saved)
  - cat score: sum softmax over ImageNet cat breeds (281–285)
  - dog score: sum softmax over ImageNet dog breeds (151–268)
  - mean and std per metric across images

Run from repo root:

    python stylegan_cats/scripts/evaluate.py
"""

import argparse
import pickle
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models import ResNet50_Weights, resnet50

SCRIPT_DIR = Path(__file__).resolve().parent
STYLEGAN_CATS_ROOT = SCRIPT_DIR.parent
REPO_ROOT = STYLEGAN_CATS_ROOT.parent

sys.path.insert(0, str(REPO_ROOT / "stylegan2-ada-pytorch"))

N_GEN = 500
BATCH_SIZE = 16
SEED = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DOG_CLASSES = list(range(151, 269))
CAT_CLASSES = list(range(281, 286))

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODELS_DIR = STYLEGAN_CATS_ROOT / "models"
OUTPUTS_DIR = STYLEGAN_CATS_ROOT / "outputs_cats"

RUN_DIR = REPO_ROOT / "stylegan_runs_cats" / (
    "00000-cats128-mirror-auto1-kimg1000-noaug"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_checkpoints() -> list[Path]:
    paths: list[Path] = []
    for base in (MODELS_DIR, RUN_DIR):
        if base.is_dir():
            paths.extend(base.glob("network-snapshot-*.pkl"))
    by_kimg: dict[int, Path] = {}
    for p in sorted(paths):
        kimg = kimg_from_path(p)
        if kimg not in by_kimg or p.parent == MODELS_DIR:
            by_kimg[kimg] = p
    return [by_kimg[k] for k in sorted(by_kimg)]


def kimg_from_path(pkl_path: Path) -> int:
    return int(pkl_path.stem.split("-")[-1])


def load_generator(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    g = data["G_ema"].to(DEVICE)
    g.eval()
    return g


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
    """images: (B, 3, H, W) in [0, 1]. Returns per-image cat and dog scores."""
    x = TF.normalize(images, IMAGENET_MEAN, IMAGENET_STD)
    probs = torch.softmax(scorer(x), dim=-1)
    cat_scores = probs[:, CAT_CLASSES].sum(dim=1).cpu().tolist()
    dog_scores = probs[:, DOG_CLASSES].sum(dim=1).cpu().tolist()
    return cat_scores, dog_scores


@torch.no_grad()
def generate_and_score(
    g,
    scorer,
    truncation_psi: float,
) -> tuple[float, float, float, float]:
    cat_scores: list[float] = []
    dog_scores: list[float] = []
    generated = 0
    batch_idx = 0

    while generated < N_GEN:
        set_seed(SEED + batch_idx)

        n_this = min(BATCH_SIZE, N_GEN - generated)
        z = torch.randn(n_this, g.z_dim, device=DEVICE)
        c = (
            torch.zeros(n_this, g.c_dim, device=DEVICE)
            if g.c_dim > 0
            else None
        )

        imgs = g(z, c, truncation_psi=truncation_psi)
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)

        batch_cat, batch_dog = species_scores_batch(scorer, imgs)
        cat_scores.extend(batch_cat)
        dog_scores.extend(batch_dog)

        generated += n_this
        batch_idx += 1
        print(f"  {generated}/{N_GEN}", end="\r")

    print()
    cat_mean, cat_std = mean_std(cat_scores)
    dog_mean, dog_std = mean_std(dog_scores)
    return cat_mean, cat_std, dog_mean, dog_std


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cat/dog ResNet50 scores for StyleGAN cat checkpoints"
        )
    )
    parser.add_argument(
        "--truncation-psi",
        type=float,
        default=1.0,
        help="StyleGAN truncation psi (default: 1.0)",
    )
    args = parser.parse_args()
    truncation_psi = args.truncation_psi

    set_seed(SEED)

    checkpoints = discover_checkpoints()
    if not checkpoints:
        raise FileNotFoundError(
            f"No network-snapshot-*.pkl in {MODELS_DIR} or {RUN_DIR}"
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    psi_tag = str(truncation_psi).replace(".", "p")
    out_csv = (
        OUTPUTS_DIR
        / f"stylegan_species_score_results_psi{psi_tag}.csv"
    )

    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"N_GEN={N_GEN}  truncation_psi={truncation_psi}")
    print(f"Cat classes: {CAT_CLASSES[0]}–{CAT_CLASSES[-1]}")
    print(f"Dog classes: {DOG_CLASSES[0]}–{DOG_CLASSES[-1]}")
    print(f"Results -> {out_csv}\n")

    with open(out_csv, "w") as f:
        f.write(
            "checkpoint,kimg,truncation_psi,"
            "cat_score,cat_score_std,dog_score,dog_score_std\n"
        )

    scorer = load_scorer()
    summary = []

    for pkl_path in checkpoints:
        kimg = kimg_from_path(pkl_path)

        print()
        print("=" * 60)
        print(f"{pkl_path.name}  (kimg={kimg})")
        print("=" * 60)

        g = load_generator(pkl_path)
        print(f"  Generating and scoring {N_GEN} images...")
        cat_mean, cat_std, dog_mean, dog_std = generate_and_score(
            g,
            scorer,
            truncation_psi,
        )
        print(
            f"  cat: mean={cat_mean:.4f}  std={cat_std:.4f}  |  "
            f"dog: mean={dog_mean:.4f}  std={dog_std:.4f}"
        )

        del g
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        summary.append(
            (pkl_path.name, kimg, cat_mean, cat_std, dog_mean, dog_std)
        )

        with open(out_csv, "a") as f:
            f.write(
                f"{pkl_path.name},{kimg},{truncation_psi},"
                f"{cat_mean:.6f},{cat_std:.6f},"
                f"{dog_mean:.6f},{dog_std:.6f}\n"
            )

    print()
    print("FINAL RESULTS (sorted by cat_score)")
    print("=" * 60)
    summary.sort(key=lambda x: x[2], reverse=True)

    for name, kimg, cat_mean, cat_std, dog_mean, dog_std in summary:
        print(
            f"  kimg={kimg:5d}  {name}  "
            f"cat={cat_mean:.4f}±{cat_std:.4f}  "
            f"dog={dog_mean:.4f}±{dog_std:.4f}"
        )

    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
