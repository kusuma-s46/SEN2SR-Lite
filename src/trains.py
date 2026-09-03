import os
import torch
import torch.nn as nn
import rasterio

from torch.utils.data import Dataset, DataLoader
from edsr_model import EDSR


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "dataset/local/train"
VAL_DIR = "dataset/local/val"

BATCH_SIZE = 1
EPOCHS = 2
LEARNING_RATE = 0.0001

MODEL_PATH = "models/edsr_local.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOCAL DATASET
# ============================================================

class LocalSRDataset(Dataset):

    def __init__(self, lr_dir, hr_dir):

        self.lr_dir = lr_dir
        self.hr_dir = hr_dir

        self.files = sorted([
            f for f in os.listdir(lr_dir)
            if f.endswith(".tif")
        ])

        print(f"Found {len(self.files)} LR-HR pairs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        lr_path = os.path.join(
            self.lr_dir,
            filename
        )

        hr_path = os.path.join(
            self.hr_dir,
            filename
        )

        # Read LR
        with rasterio.open(lr_path) as src:
            lr = src.read().astype("float32")

        # Read HR
        with rasterio.open(hr_path) as src:
            hr = src.read().astype("float32")

        # Normalize Sentinel-2 values
        lr = torch.from_numpy(lr) / 5000.0
        hr = torch.from_numpy(hr) / 5000.0

        return lr, hr


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(sr, hr):

    mse = torch.mean((sr - hr) ** 2)

    if mse == 0:
        return 100.0

    psnr = 10 * torch.log10(1.0 / mse)

    return psnr.item()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LOCAL EDSR SUPER-RESOLUTION TRAINING")
    print("=" * 60)

    print("\nUsing device:", DEVICE)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading local training dataset...")

    train_dataset = LocalSRDataset(
        TRAIN_DIR + "/lr",
        TRAIN_DIR + "/hr"
    )

    print("\nLoading local validation dataset...")

    val_dataset = LocalSRDataset(
        VAL_DIR + "/lr",
        VAL_DIR + "/hr"
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
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("\nDataLoaders created successfully!")

    # --------------------------------------------------------
    # EDSR
    # --------------------------------------------------------

    print("\nCreating EDSR model...")

    model = EDSR(
        in_channels=4,
        out_channels=4,
        num_features=64,
        num_blocks=16,
        scale=4
    ).to(DEVICE)

    print("EDSR model created successfully!")

    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )

    # --------------------------------------------------------
    # Loss + Optimizer
    # --------------------------------------------------------

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STARTING LOCAL EDSR TRAINING")
    print("=" * 60)

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

        print(f"\nEPOCH {epoch + 1}/{EPOCHS}")
        print("-" * 60)

        for batch_index, (lr, hr) in enumerate(train_loader):

            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            optimizer.zero_grad()

            sr = model(lr)

            loss = criterion(sr, hr)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            print(
                f"Training batch "
                f"{batch_index + 1}/{len(train_loader)} "
                f"- Loss: {loss.item():.6f}"
            )

        train_loss /= len(train_loader)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_psnr = 0.0

        with torch.no_grad():

            for lr, hr in val_loader:

                lr = lr.to(DEVICE)
                hr = hr.to(DEVICE)

                sr = model(lr)

                loss = criterion(sr, hr)

                val_loss += loss.item()

                val_psnr += calculate_psnr(
                    torch.clamp(sr, 0, 1),
                    hr
                )

        val_loss /= len(val_loader)

        val_psnr /= len(val_loader)

        print("\nEpoch Results")
        print("-" * 60)

        print(
            f"Training Loss:   {train_loss:.6f}"
        )

        print(
            f"Validation Loss: {val_loss:.6f}"
        )

        print(
            f"Validation PSNR: {val_psnr:.4f} dB"
        )

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------

    os.makedirs("models", exist_ok=True)

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\n")
    print("=" * 60)
    print("LOCAL EDSR TRAINING COMPLETED!")
    print("=" * 60)

    print("\nModel saved at:")
    print(MODEL_PATH)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()