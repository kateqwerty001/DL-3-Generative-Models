import csv
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class IndexDataset(Dataset):
    """
    Dataset of codebook index grids collected from a trained VQ-VAE.
    Each item is a (H, W) tensor of integer indices.
    """
    def __init__(self, indices_path):
        self.indices = torch.load(indices_path)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]


def collect_indices(vqvae, dataloader, device, save_path):
    """
    Stage 2: pass entire dataset through frozen VQ-VAE encoder
    and save codebook indices to disk.
    """
    vqvae.eval()
    all_indices = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(device)
            indices = vqvae.encode_indices(x)
            all_indices.append(indices.cpu())

    all_indices = torch.cat(all_indices, dim=0)
    torch.save(all_indices, save_path)
    print(f"Saved {len(all_indices)} index grids to {save_path}")
    return all_indices


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PixelCNNTrainer:
    """
    Trainer for PixelCNN prior (Stage 3).

    Saves after every epoch:
        - checkpoints/pixelcnn_{vqvae_name}_epoch_{N}.pt  (keeps last 3 + best)
        - logs/pixelcnn_{vqvae_name}_train_log.csv
    """

    def __init__(self, model, optimizer, device, vqvae_name="default", save_dir="checkpoints", keep_last=3):
        self.model      = model
        self.optimizer  = optimizer
        self.device     = device
        self.vqvae_name = vqvae_name
        self.save_dir   = Path(save_dir)
        self.log_dir    = Path("logs")
        self.keep_last  = keep_last

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_path     = self.log_dir / f"pixelcnn_{self.vqvae_name}_train_log.csv"
        self.best_loss    = float("inf")
        self.saved_epochs = []

        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "ce_loss", "epoch_time_sec"])

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        t0         = time.time()

        for indices in dataloader:
            indices = indices.to(self.device)

            logits = self.model(indices)

            loss = F.cross_entropy(logits, indices)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        n          = len(dataloader)
        avg_loss   = total_loss / n
        elapsed    = time.time() - t0

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(avg_loss, 6), round(elapsed, 1)])

        print(f"Epoch {epoch:3d} | ce_loss={avg_loss:.4f}  time={elapsed:.0f}s")

        self._save_checkpoint(epoch, avg_loss)
        return avg_loss

    def train(self, dataloader, num_epochs, start_epoch=1):
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.train_epoch(dataloader, epoch)

    def _save_checkpoint(self, epoch, loss):
        path = self.save_dir / f"pixelcnn_{self.vqvae_name}_epoch_{epoch:03d}.pt"
        torch.save({
            "epoch":     epoch,
            "loss":      loss,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.save_dir / f"pixelcnn_{self.vqvae_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "loss":  loss,
                "model": self.model.state_dict(),
            }, best_path)
            print(f"  -> new best saved (loss={loss:.4f})")

        self.saved_epochs.append(path)
        if len(self.saved_epochs) > self.keep_last:
            old = self.saved_epochs.pop(0)
            if old.exists():
                old.unlink()

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
        return ckpt["epoch"]