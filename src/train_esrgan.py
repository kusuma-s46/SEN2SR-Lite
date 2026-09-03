# src/train_esrgan.py

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio

from esrgan_model import ESRGAN


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_LR_DIR = "dataset/local/train/lr"
TRAIN_HR_DIR = "dataset/local/train/hr"

VAL_LR_DIR = "dataset/local/val/lr"
VAL_HR_DIR = "dataset/local/val/hr"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "esrgan_local.pth"
)

# ------------------------------------------------------------
# CPU-friendly settings
# ------------------------------------------------------------

MAX_TRAIN_SAMPLES = 50
MAX_VAL_SAMPLES = 10

BATCH_SIZE = 1

EPOCHS = 2

LEARNING_RATE = 0.0001

SCALE = 5000.0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# DATASET
# ============================================================

class SatelliteSRDataset(Dataset):

    def __init__(
        self,
        lr_dir,
        hr_dir,
        max_samples=None
    ):

        self.lr_files = sorted(
            glob.glob(
                os.path.join(
                    lr_dir,
                    "*.tif"
                )
            )
        )

        self.hr_files = sorted(
            glob.glob(
                os.path.join(
                    hr_dir,
                    "*.tif"
                )
            )
        )

        # Make sure LR and HR counts match
        if len(self.lr_files) != len(self.hr_files):

            raise ValueError(
                f"LR and HR count mismatch!\n"
                f"LR: {len(self.lr_files)}\n"
                f"HR: {len(self.hr_files)}"
            )

        if max_samples is not None:

            self.lr_files = self.lr_files[
                :max_samples
            ]

            self.hr_files = self.hr_files[
                :max_samples
            ]

    def __len__(self):

        return len(self.lr_files)

    def __getitem__(self, index):

        lr_path = self.lr_files[index]
        hr_path = self.hr_files[index]

        # ----------------------------------------------------
        # Read LR
        # ----------------------------------------------------

        with rasterio.open(lr_path) as src:

            lr = src.read().astype("float32")

        # ----------------------------------------------------
        # Read HR
        # ----------------------------------------------------

        with rasterio.open(hr_path) as src:

            hr = src.read().astype("float32")

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        lr = lr / SCALE
        hr = hr / SCALE

        # Keep values between 0 and 1
        lr = lr.clip(0.0, 1.0)
        hr = hr.clip(0.0, 1.0)

        # ----------------------------------------------------
        # Convert to tensor
        # ----------------------------------------------------

        lr = torch.from_numpy(lr).float()

        hr = torch.from_numpy(hr).float()

        return lr, hr


# ============================================================
# CREATE DATASETS
# ============================================================

print("=" * 70)
print("ESRGAN TRAINING")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

print("\nLoading training dataset...")

train_dataset = SatelliteSRDataset(
    TRAIN_LR_DIR,
    TRAIN_HR_DIR,
    MAX_TRAIN_SAMPLES
)

print(
    f"Training samples: "
    f"{len(train_dataset)}"
)


print("\nLoading validation dataset...")

val_dataset = SatelliteSRDataset(
    VAL_LR_DIR,
    VAL_HR_DIR,
    MAX_VAL_SAMPLES
)

print(
    f"Validation samples: "
    f"{len(val_dataset)}"
)


# ============================================================
# DATALOADERS
# ============================================================

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


# ============================================================
# CREATE ESRGAN MODEL
# ============================================================

print("\nCreating ESRGAN model...")

model = ESRGAN(
    in_channels=4,
    out_channels=4,
    num_features=64,
    num_blocks=6,
    growth_channels=32,
    scale=4
)

model = model.to(DEVICE)


# ============================================================
# LOSS FUNCTION
# ============================================================

# For the first CPU-friendly training stage,
# we use L1 reconstruction loss.

criterion = nn.L1Loss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...")

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Training samples: "
    f"{len(train_dataset)}"
)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    print(
        f"\nEpoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for batch_index, (lr, hr) in enumerate(
        train_loader
    ):

        lr = lr.to(DEVICE)

        hr = hr.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        sr = model(lr)

        # Calculate loss
        loss = criterion(
            sr,
            hr
        )

        # Backpropagation
        loss.backward()

        # Update model
        optimizer.step()

        running_loss += loss.item()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"\rBatch "
            f"{batch_index + 1}/"
            f"{len(train_loader)} "
            f"- Loss: "
            f"{loss.item():.6f}",
            end=""
        )

    # --------------------------------------------------------
    # Average training loss
    # --------------------------------------------------------

    average_train_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"\nTraining Loss: "
        f"{average_train_loss:.6f}"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(DEVICE)

            hr = hr.to(DEVICE)

            sr = model(lr)

            loss = criterion(
                sr,
                hr
            )

            validation_loss += loss.item()

    average_val_loss = (
        validation_loss /
        len(val_loader)
    )

    print(
        f"Validation Loss: "
        f"{average_val_loss:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving ESRGAN model...")

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(
    f"\nModel saved successfully:"
)

print(
    MODEL_PATH
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ESRGAN TRAINING COMPLETED")
print("=" * 70)

print("\nNext step:")

print(
    "Evaluate ESRGAN against "
    "Bicubic, EDSR and SwinIR."
)