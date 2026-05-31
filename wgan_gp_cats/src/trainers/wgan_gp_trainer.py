import csv
import time
from pathlib import Path

import torch
from torchvision.utils import save_image


class WGAN_GP_Trainer:

    def __init__(
            self,
            generator,
            critic,
            g_optimizer,
            c_optimizer,
            device,
            latent_dim=128,
            lambda_gp=10,
            n_critic=3,
            save_dir="checkpoints",
            sample_dir="samples",
            run_name="wgan_gp",
            keep_last=3
    ):

        self.G = generator
        self.C = critic

        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer

        self.device = device

        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic

        self.keep_last = keep_last

        self.save_dir = Path(save_dir)
        self.sample_dir = Path(sample_dir) / run_name
        self.log_dir = Path("training_logs")

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.log_dir / f"{run_name}.csv"

        self.best_g_loss = float("inf")
        self.saved_epochs = []

        self.fixed_noise = torch.randn(
            16,
            latent_dim,
            device=device
        )

        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)

                writer.writerow([
                    "epoch",
                    "critic_loss",
                    "generator_loss",
                    "wasserstein_distance",
                    "gradient_penalty",
                    "epoch_time_sec"
                ])

    def gradient_penalty(self, real, fake):

        B = real.size(0)

        alpha = torch.rand(B, 1, 1, 1, device=self.device)

        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)

        scores = self.C(interpolated)

        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(B, -1)

        gradient_norm = gradients.norm(2, dim=1)

        gp = ((gradient_norm - 1) ** 2).mean()

        return gp

    def train_epoch(self, dataloader):

        self.G.train()
        self.C.train()

        total_c_loss = 0.0
        total_g_loss = 0.0
        total_gp = 0.0
        total_wd = 0.0

        for real_images in dataloader:

            real_images = real_images.to(self.device)

            B = real_images.size(0)

            for _ in range(self.n_critic):
                z = torch.randn(B, self.latent_dim, device=self.device)

                fake_images = self.G(z).detach()

                real_score = self.C(real_images)
                fake_score = self.C(fake_images)

                wasserstein_distance = (
                        real_score.mean() - fake_score.mean()
                )

                gp = self.gradient_penalty(real_images, fake_images)

                critic_loss = (
                        -wasserstein_distance
                        + self.lambda_gp * gp
                )

                self.c_optimizer.zero_grad()

                critic_loss.backward()

                self.c_optimizer.step()

            z = torch.randn(B, self.latent_dim, device=self.device)

            fake_images = self.G(z)

            fake_score = self.C(fake_images)

            generator_loss = -fake_score.mean()

            self.g_optimizer.zero_grad()

            generator_loss.backward()

            self.g_optimizer.step()

            total_c_loss += critic_loss.item()
            total_g_loss += generator_loss.item()
            total_gp += gp.item()
            total_wd += wasserstein_distance.item()

        n = len(dataloader)

        return (
            total_c_loss / n,
            total_g_loss / n,
            total_wd / n,
            total_gp / n
        )

    @torch.no_grad()
    def save_samples(self, epoch):

        self.G.eval()

        fake_images = self.G(self.fixed_noise)

        fake_images = (fake_images + 1) / 2

        save_image(
            fake_images,
            self.sample_dir / f"epoch_{epoch:03d}.png",
            nrow=4
        )

    def train(self, train_loader, num_epochs, start_epoch=1):

        for epoch in range(start_epoch, start_epoch + num_epochs):
            t0 = time.time()

            c_loss, g_loss, wd, gp = self.train_epoch(train_loader)

            elapsed = time.time() - t0

            self.save_samples(epoch)

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)

                writer.writerow([
                    epoch,
                    round(c_loss, 6),
                    round(g_loss, 6),
                    round(wd, 6),
                    round(gp, 6),
                    round(elapsed, 1)
                ])

            print(
                f"Epoch {epoch:3d} | "
                f"C Loss: {c_loss:.4f} | "
                f"G Loss: {g_loss:.4f} | "
                f"WD: {wd:.4f} | "
                f"GP: {gp:.4f} | "
                f"Time: {elapsed:.0f}s"
            )

            self._save_checkpoint(epoch, g_loss)

    def _save_checkpoint(self, epoch, g_loss):

        path = self.save_dir / f"wgan_gp_epoch_{epoch:03d}.pt"

        torch.save({
            "epoch": epoch,
            "generator": self.G.state_dict(),
            "critic": self.C.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "c_optimizer": self.c_optimizer.state_dict(),
        }, path)

        if g_loss < self.best_g_loss:
            self.best_g_loss = g_loss

            best_path = self.save_dir / "wgan_gp_best.pt"

            torch.save({
                "epoch": epoch,
                "generator": self.G.state_dict(),
                "critic": self.C.state_dict(),
            }, best_path)

            print("  -> new best generator saved")

        self.saved_epochs.append(path)

        if len(self.saved_epochs) > self.keep_last:

            old = self.saved_epochs.pop(0)

            if old.exists():
                old.unlink()

    def load_checkpoint(self, path):

        ckpt = torch.load(path, map_location=self.device)

        self.G.load_state_dict(ckpt["generator"])
        self.C.load_state_dict(ckpt["critic"])

        self.g_optimizer.load_state_dict(ckpt["g_optimizer"])
        self.c_optimizer.load_state_dict(ckpt["c_optimizer"])

        print(f"Loaded checkpoint from {path}")

        return ckpt["epoch"]
