import os
import numpy as np
import rasterio
from tqdm import tqdm

from dataset_loader import SEN2NAIPDataset


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_SAMPLES = 100
VAL_SAMPLES = 20
TEST_SAMPLES = 20

OUTPUT_DIR = "dataset/local"


# ============================================================
# CREATE FOLDERS
# ============================================================

def create_folders():

    folders = [
        "train/lr",
        "train/hr",
        "val/lr",
        "val/hr",
        "test/lr",
        "test/hr"
    ]

    for folder in folders:
        os.makedirs(
            os.path.join(OUTPUT_DIR, folder),
            exist_ok=True
        )


# ============================================================
# SAVE ARRAY AS TIFF
# ============================================================

def save_tiff(data, path):

    data = np.asarray(data).astype(np.float32)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="float32"
    ) as dst:

        for band in range(data.shape[0]):
            dst.write(data[band], band + 1)


# ============================================================
# SAVE SAMPLES
# ============================================================

def save_samples(dataset, start_index, count, split):

    print(f"\nDownloading {split} samples...")

    for i in tqdm(range(count)):

        dataset_index = start_index + i

        try:

            lr, hr = dataset[dataset_index]

            lr = lr.numpy()
            hr = hr.numpy()

            lr_path = os.path.join(
                OUTPUT_DIR,
                split,
                "lr",
                f"{i:04d}.tif"
            )

            hr_path = os.path.join(
                OUTPUT_DIR,
                split,
                "hr",
                f"{i:04d}.tif"
            )

            save_tiff(lr, lr_path)
            save_tiff(hr, hr_path)

        except Exception as e:

            print(
                f"\nSkipping sample {dataset_index}: {e}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CREATING LOCAL SEN2NAIPv2 DATASET")
    print("=" * 60)

    create_folders()

    print("\nLoading SEN2NAIPv2...")

    dataset = SEN2NAIPDataset()

    print(
        f"Dataset contains {len(dataset)} samples."
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    save_samples(
        dataset,
        start_index=0,
        count=TRAIN_SAMPLES,
        split="train"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    save_samples(
        dataset,
        start_index=TRAIN_SAMPLES,
        count=VAL_SAMPLES,
        split="val"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    save_samples(
        dataset,
        start_index=TRAIN_SAMPLES + VAL_SAMPLES,
        count=TEST_SAMPLES,
        split="test"
    )

    print("\n" + "=" * 60)
    print("LOCAL DATASET CREATION COMPLETED")
    print("=" * 60)

    print("\nSaved inside:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()