import csv
import time
from pathlib import Path
import torch
import torch.nn.functional as F

class VQVAETrainer:
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
        self.best_val_loss  = float("inf")
        self.saved_epochs   = []

        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_recon",
                    "train_vq",
                    "train_total",
                    "val_recon",
                    "val_total",
                    "codebook_util",
                    "epoch_time_sec",
                ])

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_recon = 0.0
        total_vq    = 0.0
        all_indices = []

        for batch in train_loader:
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

        n = len(train_loader)
        utilization = self._codebook_utilization(all_indices)
        return total_recon / n, total_vq / n, (total_recon + total_vq) / n, utilization

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_recon = 0.0
        total_vq    = 0.0

        for batch in val_loader:
            x = batch.to(self.device)
            x_recon, vq_loss, _ = self.model(x)
            
            recon_loss = F.mse_loss(x_recon, x)
            
            total_recon += recon_loss.item()
            total_vq    += vq_loss.item()

        n = len(val_loader)
        return total_recon / n, total_vq / n, (total_recon + total_vq) / n

    def train(self, train_loader, val_loader, num_epochs, start_epoch=1):
        for epoch in range(start_epoch, start_epoch + num_epochs):
            t0 = time.time()
            
            # train phase
            tr_recon, tr_vq, tr_total, util = self.train_epoch(train_loader, epoch)
            
            # validation phase
            val_recon, val_vq, val_total = self.validate(val_loader)
            
            elapsed = time.time() - t0

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    round(tr_recon, 6), round(tr_vq, 6), round(tr_total, 6),
                    round(val_recon, 6), round(val_total, 6),
                    round(util, 4), round(elapsed, 1),
                ])

            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {tr_total:.4f} (R:{tr_recon:.4f}) | "
                f"Val Loss: {val_total:.4f} (R:{val_recon:.4f}) | "
                f"CB Util: {util*100:.1f}% | Time: {elapsed:.0f}s"
            )

            self._save_checkpoint(epoch, val_total)

    def _codebook_utilization(self, indices_list):
        all_idx = torch.cat([i.flatten() for i in indices_list])
        unique  = all_idx.unique().numel()
        return unique / self.model.quantizer.K

    def _save_checkpoint(self, epoch, val_loss):
        path = self.save_dir / f"vqvae_epoch_{epoch:03d}.pt"
        torch.save({
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }, path)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.save_dir / "vqvae_best.pt"
            torch.save({
                "epoch":    epoch,
                "val_loss": val_loss,
                "model":    self.model.state_dict(),
            }, best_path)
            print(f"  -> new best saved (val_loss={val_loss:.4f})")

        self.saved_epochs.append(path)
        if len(self.saved_epochs) > self.keep_last:
            old = self.saved_epochs.pop(0)
            if old.exists():
                old.unlink()