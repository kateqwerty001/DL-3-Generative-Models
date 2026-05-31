"""
Evaluate all WGAN-GP models using:

- FID
- Cat score (ResNet50 ImageNet cat classes)
- Top-100 image grid

Uses LAST checkpoint (epoch 50)
"""

import itertools
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

from PIL import Image

import torch_fidelity

sys.path.append("datasets")

from datasets.cat_datasets import get_cat_dataloaders

sys.path.append("src")

from wgan_gp_cats.src.models.wgan_gp import Generator

LAMBDA_GPS = [5, 10, 15]
N_CRITICS = [2, 3, 5]
LATENTS = [64, 128]

N_GEN = 500
N_REAL = 500

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 0

REAL_DIR = Path("outputs/fid_real")
RESULTS_PATH = Path("outputs/wgan_gp_fid_results.csv")

CAT_CLASSES = list(range(281, 286))


def load_generator(lambda_gp, n_critic, latent_dim):
    ckpt_path = (
            Path("wgan_gp_cats/scripts/checkpoints")
            / f"wgan_gp_l{lambda_gp}_n{n_critic}_z{latent_dim}"
            / "wgan_gp_epoch_050.pt"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(
        ckpt_path,
        map_location=DEVICE
    )

    G = Generator(
        latent_dim=latent_dim
    ).to(DEVICE)

    G.load_state_dict(
        ckpt["generator"]
    )

    G.eval()

    return G


def save_real_images():
    existing = list(
        REAL_DIR.glob("*.png")
    )

    if len(existing) >= N_REAL:
        print(
            f"Real images already cached "
            f"({len(existing)})"
        )
        return

    REAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    _, _, test_loader = get_cat_dataloaders(
        root_dir="data",
        batch_size=BATCH_SIZE,
        image_size=128,
        model_type="gan",
        test_size=N_REAL
    )

    saved = 0

    for batch in test_loader:

        for img in batch:

            vutils.save_image(
                img,
                REAL_DIR / f"{saved:05d}.png"
            )

            saved += 1

            if saved >= N_REAL:
                break

        if saved >= N_REAL:
            break

    print(f"Saved {saved} real images")


@torch.no_grad()
def generate_images(
        G,
        latent_dim,
        out_dir
):
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    generated = 0
    batch_idx = 0

    while generated < N_GEN:

        torch.manual_seed(SEED + batch_idx)

        n_this = min(
            BATCH_SIZE,
            N_GEN - generated
        )

        z = torch.randn(
            n_this,
            latent_dim,
            device=DEVICE
        )

        fake = G(z)

        fake = (fake + 1) / 2

        for i, img in enumerate(fake):
            vutils.save_image(
                img,
                out_dir / f"{generated + i:05d}.png"
            )

        generated += n_this
        batch_idx += 1

        print(
            f"{generated}/{N_GEN}",
            end="\r"
        )

    print()


def load_scorer():
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    model.eval()
    model.to(DEVICE)

    return model


def score_and_save_grid(
        scorer,
        image_dir,
        run_name
):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    scores = []

    files = sorted(
        image_dir.glob("*.png")
    )

    for path in files:
        img = Image.open(path).convert("RGB")

        x = TF.to_tensor(img)

        x = TF.normalize(
            x,
            mean,
            std
        )

        x = x.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = scorer(x)

            prob = (
                torch.softmax(logits, dim=-1)[0, CAT_CLASSES]
                .sum()
                .item()
            )

        scores.append(
            (prob, path)
        )

    scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    top100 = scores[:100]

    print(
        f"Top={top100[0][0]:.4f} "
        f"| 100th={top100[-1][0]:.4f}"
    )

    imgs = []

    for _, path in top100:
        img = Image.open(path).convert("RGB")

        imgs.append(
            TF.to_tensor(img)
        )

    grid = vutils.make_grid(
        torch.stack(imgs),
        nrow=10
    )

    out_grid = (
            Path("outputs")
            / f"{run_name}_top100.png"
    )

    vutils.save_image(
        grid,
        out_grid
    )

    mean_cat_score = sum(
        s[0] for s in scores
    ) / len(scores)

    return mean_cat_score


def main():
    save_real_images()

    scorer = load_scorer()

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
            RESULTS_PATH,
            "w"
    ) as f:

        f.write(
            "lambda_gp,n_critic,latent_dim,"
            "fid,cat_score\n"
        )

    results = []

    for lg, nc, ld in itertools.product(
            LAMBDA_GPS,
            N_CRITICS,
            LATENTS
    ):

        run_name = (
            f"wgan_gp_l{lg}_n{nc}_z{ld}"
        )

        print()
        print("=" * 60)

        print(run_name)
        print("=" * 60)

        try:

            G = load_generator(
                lg,
                nc,
                ld
            )

        except FileNotFoundError:

            print("Checkpoint missing")
            continue

        gen_dir = (
                Path("outputs")
                / f"generated_{run_name}"
        )

        if len(
                list(gen_dir.glob("*.png"))
        ) < N_GEN:
            print(
                f"Generating {N_GEN} images..."
            )

            generate_images(
                G,
                ld,
                gen_dir
            )

        cat_score = score_and_save_grid(
            scorer,
            gen_dir,
            run_name
        )

        print("Computing FID...")

        metrics = (
            torch_fidelity
            .calculate_metrics(
                input1=str(gen_dir),
                input2=str(REAL_DIR),
                fid=True,
                isc=False,
                kid=False,
                verbose=False,
            )
        )

        fid = metrics[
            "frechet_inception_distance"
        ]

        print(f"FID = {fid:.3f}")

        results.append(
            (
                lg,
                nc,
                ld,
                fid,
                cat_score
            )
        )

        with open(
                RESULTS_PATH,
                "a"
        ) as f:

            f.write(
                f"{lg},{nc},{ld},"
                f"{fid:.4f},"
                f"{cat_score:.6f}\n"
            )

        del G
        torch.cuda.empty_cache()

    print("\nFINAL RESULTS")

    results.sort(
        key=lambda x: x[3]
    )

    for r in results:
        lg, nc, ld, fid, cs = r

        print(
            f"λ={lg:2d} "
            f"n={nc} "
            f"z={ld:3d} "
            f"FID={fid:.2f} "
            f"CatScore={cs:.4f}"
        )


if __name__ == "__main__":
    main()
