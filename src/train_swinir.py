import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio

from swinir_model import SwinIR


# ============================================================
# Configuration
# ============================================================

TRAIN_LR_DIR = "dataset/local/train/lr"
TRAIN_HR_DIR = "dataset/local/train/hr"

VAL_LR_DIR = "dataset/local/val/lr"
VAL_HR_DIR = "dataset/local/val/hr"

MODEL_SAVE_PATH = "models/swinir_local.pth"

BATCH_SIZE = 1
EPOCHS = 2
LEARNING_RATE = 0.0001

MAX_TRAIN_SAMPLES = 100
MAX_VAL_SAMPLES = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Dataset
# ============================================================

class LocalSatelliteDataset(Dataset):

    def __init__(self, lr_dir, hr_dir, max_samples=None):

        self.lr_files = sorted(
            glob.glob(os.path.join(lr_dir, "*.tif"))
        )

        self.hr_files = sorted(
            glob.glob(os.path.join(hr_dir, "*.tif"))
        )

        if len(self.lr_files) != len(self.hr_files):
            raise ValueError(
                "Number of LR and HR images does not match."
            )

        if max_samples is not None:
            self.lr_files = self.lr_files[:max_samples]
            self.hr_files = self.hr_files[:max_samples]

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, index):

        lr_path = self.lr_files[index]
        hr_path = self.hr_files[index]

        with rasterio.open(lr_path) as src:
            lr = src.read()

        with rasterio.open(hr_path) as src:
            hr = src.read()

        lr = torch.tensor(
            lr,
            dtype=torch.float32
        )

        hr = torch.tensor(
            hr,
            dtype=torch.float32
        )

        # Normalize satellite pixel values
        lr = lr / 5000.0
        hr = hr / 5000.0

        return lr, hr


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(predicted, target):

    mse = torch.mean(
        (predicted - target) ** 2
    )

    if mse == 0:
        return 100.0

    psnr = 10 * torch.log10(
        1.0 / mse
    )

    return psnr.item()


# ============================================================
# Validation
# ============================================================

def validate(model, val_loader):

    model.eval()

    total_psnr = 0.0
    total_samples = 0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            sr = model(lr)

            # Keep output within normalized range
            sr = torch.clamp(
                sr,
                0.0,
                1.0
            )

            psnr = calculate_psnr(
                sr,
                hr
            )

            total_psnr += psnr
            total_samples += 1

    if total_samples == 0:
        return 0.0

    return total_psnr / total_samples


# ============================================================
# Main Training
# ============================================================

def main():

    print("=" * 60)
    print("SwinIR Multispectral Super-Resolution Training")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    train_dataset = LocalSatelliteDataset(
        TRAIN_LR_DIR,
        TRAIN_HR_DIR,
        MAX_TRAIN_SAMPLES
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print("\nLoading validation dataset...")

    val_dataset = LocalSatelliteDataset(
        VAL_LR_DIR,
        VAL_HR_DIR,
        MAX_VAL_SAMPLES
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating SwinIR model...")

    model = SwinIR(
        in_channels=4,
        out_channels=4,
        embed_dim=64,
        depths=4,
        num_heads=4,
        window_size=8,
        upscale=4
    ).to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.L1Loss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting training...")
    print("-" * 60)

    best_psnr = 0.0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for batch_index, (lr, hr) in enumerate(train_loader):

            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            sr = model(lr)

            # Keep prediction in valid normalized range
            sr = torch.clamp(
                sr,
                0.0,
                1.0
            )

            # Calculate loss
            loss = criterion(
                sr,
                hr
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

            print(
                f"\rEpoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}",
                end=""
            )

        average_loss = (
            running_loss /
            len(train_loader)
        )

        print()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_psnr = validate(
            model,
            val_loader
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Average Loss: {average_loss:.6f} "
            f"Validation PSNR: {val_psnr:.4f} dB"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_psnr > best_psnr:

            best_psnr = val_psnr

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print(
                "Best SwinIR model saved."
            )

    print("\n" + "=" * 60)
    print("Training completed.")
    print("=" * 60)

    print(
        "\nBest Validation PSNR:",
        f"{best_psnr:.4f} dB"
    )

    print(
        "\nModel saved at:",
        MODEL_SAVE_PATH
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()