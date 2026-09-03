import torch
from torch.utils.data import DataLoader, Subset

from dataset_loader import SEN2NAIPDataset


# ============================================================
# NORMALIZATION SETTINGS
# ============================================================

# Initial working scale based on the observed SEN2NAIPv2 values.
# We will verify/refine this later using dataset statistics.
NORMALIZATION_SCALE = 5000.0


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize_image(image):
    """
    Normalize satellite image values to approximately [0, 1].

    Input:
        [C, H, W]
        or
        [B, C, H, W]

    Output:
        Same shape, normalized to [0, 1].
    """

    image = image / NORMALIZATION_SCALE

    image = torch.clamp(
        image,
        min=0.0,
        max=1.0
    )

    return image


# ============================================================
# PREPROCESS LR + HR
# ============================================================

def preprocess_pair(lr_image, hr_image):

    # Use exactly the same normalization
    # for LR and HR.

    lr_image = normalize_image(
        lr_image
    )

    hr_image = normalize_image(
        hr_image
    )

    return lr_image, hr_image


# ============================================================
# CREATE DATASET SPLIT
# ============================================================

def create_datasets(dataset):

    total_samples = len(dataset)

    # 80% training
    # 10% validation
    # 10% testing

    train_size = int(
        0.80 * total_samples
    )

    val_size = int(
        0.10 * total_samples
    )

    test_size = (
        total_samples
        - train_size
        - val_size
    )

    # Reproducible split
    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        total_samples,
        generator=generator
    )

    train_indices = indices[
        :train_size
    ]

    val_indices = indices[
        train_size:
        train_size + val_size
    ]

    test_indices = indices[
        train_size + val_size:
    ]

    train_dataset = Subset(
        dataset,
        train_indices
    )

    val_dataset = Subset(
        dataset,
        val_indices
    )

    test_dataset = Subset(
        dataset,
        test_indices
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SEN2NAIPv2 DATA PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = SEN2NAIPDataset()

    print("\nDataset loaded.")
    print(
        "Total samples:",
        len(dataset)
    )

    # --------------------------------------------------------
    # Test preprocessing
    # --------------------------------------------------------

    print(
        "\nTesting preprocessing..."
    )

    lr, hr = dataset[0]

    print("\nOriginal data")
    print("-" * 30)

    print(
        "LR shape:",
        lr.shape
    )

    print(
        "HR shape:",
        hr.shape
    )

    print(
        "LR min:",
        lr.min().item()
    )

    print(
        "LR max:",
        lr.max().item()
    )

    print(
        "HR min:",
        hr.min().item()
    )

    print(
        "HR max:",
        hr.max().item()
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    lr_processed, hr_processed = (
        preprocess_pair(
            lr,
            hr
        )
    )

    print("\nNormalized data")
    print("-" * 30)

    print(
        "LR shape:",
        lr_processed.shape
    )

    print(
        "HR shape:",
        hr_processed.shape
    )

    print(
        "LR min:",
        lr_processed.min().item()
    )

    print(
        "LR max:",
        lr_processed.max().item()
    )

    print(
        "HR min:",
        hr_processed.min().item()
    )

    print(
        "HR max:",
        hr_processed.max().item()
    )

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    print(
        "\nCreating dataset splits..."
    )

    train_dataset, val_dataset, test_dataset = (
        create_datasets(dataset)
    )

    print("\nDataset split completed!")
    print("-" * 30)

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    print(
        "Testing samples:",
        len(test_dataset)
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print(
        "\nCreating DataLoaders..."
    )

    # CPU system → keep batch size small
    batch_size = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(
        "DataLoaders created successfully!"
    )

    # --------------------------------------------------------
    # Test one batch
    # --------------------------------------------------------

    print(
        "\nTesting one training batch..."
    )

    lr_batch, hr_batch = next(
        iter(train_loader)
    )

    # Normalize the batch
    lr_batch = normalize_image(
        lr_batch
    )

    hr_batch = normalize_image(
        hr_batch
    )

    print("\nTraining batch")
    print("-" * 30)

    print(
        "LR batch shape:",
        lr_batch.shape
    )

    print(
        "HR batch shape:",
        hr_batch.shape
    )

    print(
        "LR batch min:",
        lr_batch.min().item()
    )

    print(
        "LR batch max:",
        lr_batch.max().item()
    )

    print(
        "HR batch min:",
        hr_batch.min().item()
    )

    print(
        "HR batch max:",
        hr_batch.max().item()
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "DATA PREPROCESSING COMPLETED SUCCESSFULLY!"
    )
    print("=" * 60)

    print("\nNext step:")
    print(
        "EDSR MODEL TRAINING"
    )