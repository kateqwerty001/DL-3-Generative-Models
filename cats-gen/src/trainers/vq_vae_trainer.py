import csv
import time
from pathlib import Path
import torch
import torch.nn.functional as F


class VQVAETrainer:
    """
    Trainer for VQ-VAE Stage 1: Encoder + VectorQuantizer + Decoder.

    Saves after every epoch:
        - checkpoints/vqvae_epoch_{N}.pt   (keeps last 3 + best)
        - logs/vqvae_train_log.csv         (one row per epoch)
    """

    def __init__(self, model, optimizer, device, save_dir="checkpoints", keep_last=3, run_name="vqvae"):
        self.model      = model
        self.optimizer  = optimizer
        self.device     = device
        self.save_dir   = Path(save_dir)
        self.log_dir    = Path("logs")
        self.keep_last  = keep_last

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_path       = self.log_dir / f"{run_name}_train_log.csv"
        self.best_loss      = float("inf")
        self.saved_epochs   = []

        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "recon_loss",
                    "vq_loss",
                    "total_loss",
                    "codebook_utilization",
                    "epoch_time_sec",
                ])

    # ------------------------------------------------------------------

    def train_epoch(self, dataloader, epoch):
        self.model.train()

        total_recon = 0.0
        total_vq    = 0.0
        all_indices = []
        t0          = time.time()

        for batch in dataloader:
            x = batch.to(self.device)

            x_recon, vq_loss, indices = self.model(x)

            recon_loss  = F.mse_loss(x_recon, x)
            total_loss  = recon_loss + vq_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            total_recon += recon_loss.item()
            total_vq    += vq_loss.item()
            all_indices.append(indices.detach().cpu())

        n               = len(dataloader)
        avg_recon       = total_recon / n
        avg_vq          = total_vq / n
        avg_total       = avg_recon + avg_vq
        utilization     = self._codebook_utilization(all_indices)
        elapsed         = time.time() - t0

        # log to CSV
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(avg_recon,   6),
                round(avg_vq,      6),
                round(avg_total,   6),
                round(utilization, 4),
                round(elapsed,     1),
            ])

        print(
            f"Epoch {epoch:3d} | "
            f"recon={avg_recon:.4f}  vq={avg_vq:.4f}  "
            f"total={avg_total:.4f}  "
            f"codebook={utilization*100:.1f}%  "
            f"time={elapsed:.0f}s"
        )

        self._save_checkpoint(epoch, avg_total)

        return avg_recon, avg_vq, avg_total, utilization

    def train(self, dataloader, num_epochs, start_epoch=1):
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.train_epoch(dataloader, epoch)

    def _codebook_utilization(self, indices_list):
        all_idx = torch.cat([i.flatten() for i in indices_list])
        unique  = all_idx.unique().numel()
        return unique / self.model.quantizer.K

    def _save_checkpoint(self, epoch, loss):
        path = self.save_dir / f"vqvae_epoch_{epoch:03d}.pt"
        torch.save({
            "epoch":      epoch,
            "loss":       loss,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }, path)

        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.save_dir / "vqvae_best.pt"
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