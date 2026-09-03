"""
RGB-EDSR Training
SEN2SR-Lite RGB Demonstration Branch

Input:
    Sentinel-2 B04, B03, B02 -> RGB

LR:
    130 x 130

HR:
    520 x 520

Scale:
    4x

Output model:
    models/rgb_edsr_local.pth
"""

import os
import glob
import numpy as np
import rasterio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rgb_edsr_model import RGB_EDSR


# ============================================================
# PROJECT PATHS
# ============================================================

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SRC_DIR
)

TRAIN_LR_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "local",
    "train",
    "lr"
)

TRAIN_HR_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "local",
    "train",
    "hr"
)

VAL_LR_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "local",
    "val",
    "lr"
)

VAL_HR_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "local",
    "val",
    "hr"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "rgb_edsr_local.pth"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

MAX_TRAIN_SAMPLES = 100
MAX_VAL_SAMPLES = 20

BATCH_SIZE = 1

EPOCHS = 2

LEARNING_RATE = 0.0001

MAX_VALUE = 5000.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class RGBSatelliteDataset(Dataset):

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

        # ----------------------------------------------------
        # Match LR and HR files
        # ----------------------------------------------------

        lr_names = {
            os.path.basename(path): path
            for path in self.lr_files
        }

        hr_names = {
            os.path.basename(path): path
            for path in self.hr_files
        }

        common_names = sorted(
            set(lr_names.keys())
            &
            set(hr_names.keys())
        )

        self.lr_files = [
            lr_names[name]
            for name in common_names
        ]

        self.hr_files = [
            hr_names[name]
            for name in common_names
        ]

        # ----------------------------------------------------
        # Limit samples
        # ----------------------------------------------------

        if max_samples is not None:

            self.lr_files = self.lr_files[
                :max_samples
            ]

            self.hr_files = self.hr_files[
                :max_samples
            ]

        print(
            f"Dataset: {len(self.lr_files)} pairs"
        )

        # ----------------------------------------------------
        # Stop if no data
        # ----------------------------------------------------

        if len(self.lr_files) == 0:

            raise RuntimeError(
                "\nNo matching LR-HR .tif pairs found.\n\n"
                f"LR folder:\n{lr_dir}\n\n"
                f"HR folder:\n{hr_dir}\n\n"
                "Please check that both folders contain "
                "matching .tif files."
            )

    def __len__(self):

        return len(self.lr_files)

    def __getitem__(self, index):

        lr_path = self.lr_files[index]
        hr_path = self.hr_files[index]

        # ----------------------------------------------------
        # Read LR
        # ----------------------------------------------------

        with rasterio.open(lr_path) as src:

            lr = src.read().astype(
                np.float32
            )

        # ----------------------------------------------------
        # Read HR
        # ----------------------------------------------------

        with rasterio.open(hr_path) as src:

            hr = src.read().astype(
                np.float32
            )

        # ----------------------------------------------------
        # Check bands
        # ----------------------------------------------------

        if lr.shape[0] < 4:

            raise ValueError(
                f"LR image has fewer than 4 bands:\n"
                f"{lr_path}"
            )

        if hr.shape[0] < 4:

            raise ValueError(
                f"HR image has fewer than 4 bands:\n"
                f"{hr_path}"
            )

        # ----------------------------------------------------
        # Sentinel-2 bands
        #
        # Original:
        # Band 0 = B02 Blue
        # Band 1 = B03 Green
        # Band 2 = B04 Red
        # Band 3 = B08 NIR
        #
        # RGB:
        # Red   = B04
        # Green = B03
        # Blue  = B02
        # ----------------------------------------------------

        lr_rgb = lr[
            [2, 1, 0],
            :, :
        ]

        hr_rgb = hr[
            [2, 1, 0],
            :, :
        ]

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        lr_rgb = np.clip(
            lr_rgb / MAX_VALUE,
            0.0,
            1.0
        )

        hr_rgb = np.clip(
            hr_rgb / MAX_VALUE,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        lr_tensor = torch.from_numpy(
            lr_rgb
        ).float()

        hr_tensor = torch.from_numpy(
            hr_rgb
        ).float()

        return lr_tensor, hr_tensor


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    prediction,
    target
):

    mse = torch.mean(
        (prediction - target) ** 2
    )

    if mse.item() == 0:

        return 100.0

    psnr = 10 * torch.log10(
        1.0 / mse
    )

    return psnr.item()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("RGB-EDSR TRAINING")
    print("=" * 70)

    print(
        f"\nProject root:\n{PROJECT_ROOT}"
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        "Input: RGB Sentinel-2 bands "
        "(B04, B03, B02)"
    )

    print(
        "Scale: 4x"
    )

    print(
        "LR: 130x130"
    )

    print(
        "HR: 520x520"
    )

    # ========================================================
    # CHECK DATASET PATHS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CHECKING DATASET PATHS")
    print("=" * 70)

    print(
        f"\nTrain LR:\n{TRAIN_LR_DIR}"
    )

    print(
        f"\nTrain HR:\n{TRAIN_HR_DIR}"
    )

    print(
        f"\nVal LR:\n{VAL_LR_DIR}"
    )

    print(
        f"\nVal HR:\n{VAL_HR_DIR}"
    )

    paths = [
        TRAIN_LR_DIR,
        TRAIN_HR_DIR,
        VAL_LR_DIR,
        VAL_HR_DIR
    ]

    for path in paths:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nFolder not found:\n{path}"
            )

    print(
        "\n[OK] All dataset folders found."
    )

    # ========================================================
    # COUNT FILES
    # ========================================================

    train_lr_count = len(
        glob.glob(
            os.path.join(
                TRAIN_LR_DIR,
                "*.tif"
            )
        )
    )

    train_hr_count = len(
        glob.glob(
            os.path.join(
                TRAIN_HR_DIR,
                "*.tif"
            )
        )
    )

    val_lr_count = len(
        glob.glob(
            os.path.join(
                VAL_LR_DIR,
                "*.tif"
            )
        )
    )

    val_hr_count = len(
        glob.glob(
            os.path.join(
                VAL_HR_DIR,
                "*.tif"
            )
        )
    )

    print("\nDataset file counts:")

    print(
        f"Train LR: {train_lr_count}"
    )

    print(
        f"Train HR: {train_hr_count}"
    )

    print(
        f"Val LR:   {val_lr_count}"
    )

    print(
        f"Val HR:   {val_hr_count}"
    )

    # ========================================================
    # CREATE DATASETS
    # ========================================================

    print("\n")
    print("Loading training dataset...")

    train_dataset = RGBSatelliteDataset(
        TRAIN_LR_DIR,
        TRAIN_HR_DIR,
        MAX_TRAIN_SAMPLES
    )

    print(
        "\nLoading validation dataset..."
    )

    val_dataset = RGBSatelliteDataset(
        VAL_LR_DIR,
        VAL_HR_DIR,
        MAX_VAL_SAMPLES
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

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

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CREATING RGB-EDSR MODEL")
    print("=" * 70)

    model = RGB_EDSR(
        scale=4,
        num_features=64,
        num_blocks=16
    ).to(DEVICE)

    print(
        "\n[OK] RGB-EDSR model created."
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.L1Loss()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # CREATE MODEL DIRECTORY
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STARTING RGB-EDSR TRAINING")
    print("=" * 70)

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print("-" * 70)

        for batch_index, (
            lr,
            hr
        ) in enumerate(train_loader):

            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            # Clear gradients

            optimizer.zero_grad()

            # Forward pass

            sr = model(lr)

            # Loss

            loss = criterion(
                sr,
                hr
            )

            # Backpropagation

            loss.backward()

            # Update model weights

            optimizer.step()

            total_loss += loss.item()

            if (
                batch_index + 1
            ) % 10 == 0:

                print(
                    f"Batch "
                    f"{batch_index + 1}/"
                    f"{len(train_loader)} | "
                    f"Loss: "
                    f"{loss.item():.6f}"
                )

        # ====================================================
        # AVERAGE TRAINING LOSS
        # ====================================================

        average_loss = (
            total_loss /
            len(train_loader)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        validation_psnr = []

        with torch.no_grad():

            for lr, hr in val_loader:

                lr = lr.to(DEVICE)
                hr = hr.to(DEVICE)

                sr = model(lr)

                sr = torch.clamp(
                    sr,
                    0.0,
                    1.0
                )

                psnr = calculate_psnr(
                    sr,
                    hr
                )

                validation_psnr.append(
                    psnr
                )

        average_psnr = np.mean(
            validation_psnr
        )

        # ====================================================
        # EPOCH RESULT
        # ====================================================

        print("\nEpoch result:")

        print(
            f"Training Loss : "
            f"{average_loss:.6f}"
        )

        print(
            f"Validation PSNR: "
            f"{average_psnr:.4f} dB"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RGB-EDSR TRAINING COMPLETED")
    print("=" * 70)

    print(
        "\nModel saved successfully:"
    )

    print(
        MODEL_PATH
    )

    print("\nModel configuration:")

    print(
        "Input channels : 3"
    )

    print(
        "Output channels: 3"
    )

    print(
        "Scale          : 4x"
    )

    print(
        "Features       : 64"
    )

    print(
        "Residual blocks: 16"
    )

    print("\nRGB branch is ready for inference.")

    print("=" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()