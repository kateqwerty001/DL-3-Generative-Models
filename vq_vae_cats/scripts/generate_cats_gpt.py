"""
Compute FID for all 9 VQ-VAE + GPT prior combinations.
Generates N_GEN images per model, compares against N_REAL real test images.
Saves a 10x10 grid of top-100 images (by cat score) for each model.
Results saved to outputs/fid_results.csv
"""

import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torch_fidelity

sys.path.append("src")

from models.vq_vae    import VQ_VAE
from models.gpt_prior import GPTPrior
from cat_datasets     import get_cat_dataloaders

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────

BETAS       = [0.5, 1.0, 1.5]
KS          = [128, 256, 512]

N_GEN       = 500
N_REAL      = 500
BATCH_SIZE  = 16
SEED        = 0
TEMPERATURE = 0.85

LATENT_H   = 32
LATENT_W   = 32
HIDDEN_DIM = 256
N_LAYERS   = 8
N_HEADS    = 8

# ImageNet indices for cat breeds (tabby, tiger cat, Persian, Siamese, Egyptian)
CAT_CLASSES = list(range(281, 286))

REAL_DIR  = Path("outputs/fid_real")
LOG_PATH  = Path(f"outputs/fid_results_{TEMPERATURE}.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────

def load_vqvae(beta, K):
    path = f"vqvae_models/checkpoints_{beta}_{K}/vqvae_best.pt"
    ckpt  = torch.load(path, map_location=DEVICE)
    model = VQ_VAE(in_channels=3, hidden_dim=128, embedding_dim=256,
                   num_embedding=K, n_residual=2, beta=beta).to(DEVICE)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model


def load_gpt(beta, K):
    path = f"checkpoints/gpt_{beta}_{K}/best.pt"
    ckpt  = torch.load(path, map_location=DEVICE)
    model = GPTPrior(num_embeddings=K, latent_h=LATENT_H, latent_w=LATENT_W,
                     hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                     n_heads=N_HEADS).to(DEVICE)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model


def load_scorer():
    """ResNet50 pretrained on ImageNet for cat scoring."""
    scorer = resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)
    scorer.eval()
    return scorer


def generate_images(vqvae, gpt, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    batch_idx = 0
    while generated < N_GEN:
        torch.manual_seed(SEED + batch_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + batch_idx)
        n_this  = min(BATCH_SIZE, N_GEN - generated)
        indices = gpt.generate((n_this, LATENT_H, LATENT_W), DEVICE, TEMPERATURE)
        with torch.no_grad():
            images = vqvae.decode_indices(indices)
        for i, img in enumerate(images):
            vutils.save_image(img, out_dir / f"{generated + i:05d}.png")
        generated += n_this
        batch_idx += 1
        print(f"    {generated}/{N_GEN}", end="\r")
    print()


def score_and_save_grid(scorer, gen_dir, beta, K):
    """Score all generated images by cat probability, save top-100 grid."""
    print("  Scoring images...")

    # ResNet expects ImageNet normalization
    normalize = TF.normalize
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    scores = []
    all_files = sorted(gen_dir.glob("*.png"))

    for img_path in all_files:
        img = Image.open(img_path).convert("RGB")
        t   = TF.to_tensor(img)
        t   = normalize(t, mean, std).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = scorer(t)
            prob   = torch.softmax(logits, dim=-1)[0, CAT_CLASSES].sum().item()
        scores.append((prob, img_path))

    # sort by cat score descending, take top 100
    scores.sort(key=lambda x: x[0], reverse=True)
    top100 = scores[:100]

    print(f"  Top cat score: {top100[0][0]:.4f}  |  "
          f"100th score: {top100[-1][0]:.4f}  |  "
          f"worst score: {scores[-1][0]:.4f}")

    # build 10x10 grid
    grid_imgs = []
    for _, img_path in top100:
        img = Image.open(img_path).convert("RGB")
        grid_imgs.append(TF.to_tensor(img))

    grid = vutils.make_grid(torch.stack(grid_imgs), nrow=10, padding=2)
    grid_path = f"outputs/grid_beta{beta}_K{K}_{TEMPERATURE}.png"
    vutils.save_image(grid, grid_path)
    print(f"  Top-100 grid saved to {grid_path}")


def save_real_images():
    existing = list(REAL_DIR.glob("*.png"))
    if len(existing) >= N_REAL:
        print(f"  Real images already cached ({len(existing)}), skipping.")
        return
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    _, _, test_loader = get_cat_dataloaders(
        "data", batch_size=BATCH_SIZE, image_size=128,
        num_workers=1, model_type="vqvae",
        val_size=500, test_size=N_REAL, seed=42,
    )
    saved = 0
    for batch in test_loader:
        for img in batch:
            vutils.save_image(img, REAL_DIR / f"{saved:05d}.png")
            saved += 1
            if saved >= N_REAL:
                break
        if saved >= N_REAL:
            break
    print(f"  Saved {saved} real images to {REAL_DIR}")


# ────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Grid: {len(BETAS)} x {len(KS)} = {len(BETAS)*len(KS)} models")
    print(f"N_GEN={N_GEN}  N_REAL={N_REAL}  temperature={TEMPERATURE}\n")

    if not LOG_PATH.exists():
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w") as f:
            f.write("beta,K,FID\n")

    print("Preparing real images...")
    save_real_images()

    print("Loading scorer (ResNet50)...")
    scorer = load_scorer()

    results = []

    for beta in BETAS:
        for K in KS:
            print(f"\n{'='*50}")
            print(f"  beta={beta}  K={K}")
            print(f"{'='*50}")

            gen_dir  = Path(f"outputs/fid_generated_beta{beta}_K{K}_{TEMPERATURE}")
            existing = list(gen_dir.glob("*.png"))

            if len(existing) >= N_GEN:
                print(f"  Generated images already cached ({len(existing)}), skipping.")
            else:
                try:
                    vqvae = load_vqvae(beta, K)
                    gpt   = load_gpt(beta, K)
                    print(f"  Generating {N_GEN} images...")
                    generate_images(vqvae, gpt, gen_dir)
                    del vqvae, gpt
                    torch.cuda.empty_cache()
                except FileNotFoundError as e:
                    print(f"  SKIPPED — checkpoint not found: {e}")
                    continue

            # top-100 grid
            score_and_save_grid(scorer, gen_dir, beta, K)

            # FID
            print("  Computing FID...")
            metrics = torch_fidelity.calculate_metrics(
                input1=str(gen_dir),
                input2=str(REAL_DIR),
                fid=True,
                isc=False,
                kid=False,
                verbose=False,
            )
            fid = metrics["frechet_inception_distance"]
            print(f"  FID = {fid:.2f}")

            results.append((beta, K, fid))
            with open(LOG_PATH, "a") as f:
                f.write(f"{beta},{K},{fid:.4f}\n")

    print(f"\n{'='*50}")
    print("  FINAL RESULTS (sorted by FID)")
    print(f"{'='*50}")
    results.sort(key=lambda x: x[2])
    for beta, K, fid in results:
        print(f"  beta={beta}  K={K:4d}  FID={fid:.2f}")
    print(f"\nAll results saved to {LOG_PATH}")


if __name__ == "__main__":
    main()