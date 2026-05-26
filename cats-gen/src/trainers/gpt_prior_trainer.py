import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class IndexDataset(Dataset):
    """Dataset of codebook index grids (H, W) collected from a frozen VQ-VAE."""

    def __init__(self, indices_path):
        self.indices = torch.load(indices_path)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]


class GPTPriorTrainer:
    """
    Trainer for GPTPrior.

    Saves:
        checkpoints/{run_name}/epoch_{N:03d}.pt   (keeps last keep_last)
        checkpoints/{run_name}/best.pt
        logs/{run_name}_train_log.csv
    """

    def __init__(self, model, optimizer, device,
                 run_name="gpt_prior", save_dir="checkpoints", keep_last=3):
        self.model     = model
        self.optimizer = optimizer
        self.device    = device
        self.run_name  = run_name
        self.save_dir  = Path(save_dir) / run_name
        self.log_dir   = Path("logs")
        self.keep_last = keep_last

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_path     = self.log_dir / f"{run_name}_train_log.csv"
        self.best_val     = float("inf")
        self.saved_epochs = []

        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["epoch", "train_ce", "val_ce", "epoch_time_sec"]
                )

    def train_epoch(self, train_loader):
        self.model.train()
        total = 0.0
        for indices in train_loader:
            indices = indices.to(self.device)
            logits  = self.model(indices)                  # (B, K, H, W)
            loss    = F.cross_entropy(logits, indices)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total = 0.0
        for indices in val_loader:
            indices = indices.to(self.device)
            logits  = self.model(indices)
            total  += F.cross_entropy(logits, indices).item()
        return total / len(val_loader)

    def train(self, train_loader, val_loader, max_epochs, patience, start_epoch=1):
        no_improve = 0

        for epoch in range(start_epoch, start_epoch + max_epochs):
            t0       = time.time()
            train_ce = self.train_epoch(train_loader)
            val_ce   = self.validate(val_loader)
            elapsed  = time.time() - t0

            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, round(train_ce, 6), round(val_ce, 6), round(elapsed, 1)]
                )

            print(f"Epoch {epoch:3d} | train_ce={train_ce:.4f}  val_ce={val_ce:.4f}  "
                  f"best={self.best_val:.4f}  patience={no_improve}/{patience}  "
                  f"time={elapsed:.0f}s")

            if val_ce < self.best_val:
                self.best_val = val_ce
                no_improve    = 0
                self._save(epoch, val_ce, best=True)
                print(f"  -> new best saved (val_ce={val_ce:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            self._save(epoch, val_ce, best=False)

        return self.best_val

    def _save(self, epoch, val_ce, best=False):
        payload = {
            "epoch":   epoch,
            "val_ce":  val_ce,
            "model":   self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if best:
            torch.save(payload, self.save_dir / "best.pt")
        else:
            path = self.save_dir / f"epoch_{epoch:03d}.pt"
            torch.save(payload, path)
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