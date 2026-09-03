import torch
from torch.utils.data import Dataset, DataLoader
import tacoreader.v1 as tacoreader
import rasterio


class SEN2NAIPDataset(Dataset):

    def __init__(self):

        print("Connecting to SEN2NAIPv2...")

        self.dataset = tacoreader.load(
            "tacofoundation:sen2naipv2-unet"
        )

        print("Dataset loaded successfully!")
        print(
            "Number of samples:",
            len(self.dataset)
        )

    def __len__(self):

        return len(self.dataset)

    def _read_sample(self, index):

        # IMPORTANT:
        # PyTorch Subset can provide a tensor index.
        # tacoreader requires a normal Python integer.
        index = int(index)

        sample = self.dataset.read(index)

        # ----------------------------------------------------
        # Find LR and HR paths
        # ----------------------------------------------------

        lr_rows = sample.loc[
            sample["tortilla:id"] == "lr",
            "internal:subfile"
        ]

        hr_rows = sample.loc[
            sample["tortilla:id"] == "hr",
            "internal:subfile"
        ]

        if len(lr_rows) == 0:

            raise ValueError(
                f"LR image not found for sample {index}"
            )

        if len(hr_rows) == 0:

            raise ValueError(
                f"HR image not found for sample {index}"
            )

        lr_path = lr_rows.iloc[0]
        hr_path = hr_rows.iloc[0]

        # ----------------------------------------------------
        # Read LR image
        # ----------------------------------------------------

        with rasterio.open(lr_path) as src:

            lr_image = src.read()

        # ----------------------------------------------------
        # Read HR image
        # ----------------------------------------------------

        with rasterio.open(hr_path) as src:

            hr_image = src.read()

        # ----------------------------------------------------
        # Convert NumPy → PyTorch
        # ----------------------------------------------------

        lr_image = torch.from_numpy(
            lr_image
        ).float()

        hr_image = torch.from_numpy(
            hr_image
        ).float()

        # ----------------------------------------------------
        # Check number of bands
        # ----------------------------------------------------

        if lr_image.shape[0] != 4:

            raise ValueError(
                f"LR has {lr_image.shape[0]} bands. "
                f"Expected 4."
            )

        if hr_image.shape[0] != 4:

            raise ValueError(
                f"HR has {hr_image.shape[0]} bands. "
                f"Expected 4."
            )

        # ----------------------------------------------------
        # Check ×4 spatial relationship
        # ----------------------------------------------------

        expected_height = (
            lr_image.shape[1] * 4
        )

        expected_width = (
            lr_image.shape[2] * 4
        )

        if (
            hr_image.shape[1] != expected_height
            or
            hr_image.shape[2] != expected_width
        ):

            raise ValueError(
                f"LR/HR size mismatch.\n"
                f"LR: {lr_image.shape}\n"
                f"HR: {hr_image.shape}"
            )

        return lr_image, hr_image

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        # Convert PyTorch tensor index to Python int
        index = int(index)

        try:

            return self._read_sample(index)

        except Exception as error:

            print(
                f"\nWarning: Could not read sample {index}"
            )

            print(
                f"Reason: {type(error).__name__}: {error}"
            )

            # ------------------------------------------------
            # Try nearby samples
            # ------------------------------------------------

            for offset in range(1, 21):

                next_index = (
                    index + offset
                ) % len(self.dataset)

                try:

                    print(
                        f"Trying sample {next_index}..."
                    )

                    lr_image, hr_image = (
                        self._read_sample(
                            next_index
                        )
                    )

                    print(
                        f"Successfully loaded "
                        f"sample {next_index}."
                    )

                    return lr_image, hr_image

                except Exception:

                    continue

            # ------------------------------------------------
            # No valid sample found
            # ------------------------------------------------

            raise RuntimeError(
                f"Could not find a valid sample "
                f"near index {index}."
            )


# ============================================================
# TEST DATASET
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING SEN2NAIPv2 DATASET")
    print("=" * 60)

    dataset = SEN2NAIPDataset()

    print("\nTesting sample 0...")

    lr, hr = dataset[0]

    print("\nDataset test completed!")

    print("\nLR information")
    print("-" * 30)

    print("Shape:", lr.shape)
    print("Data type:", lr.dtype)
    print("Minimum:", lr.min().item())
    print("Maximum:", lr.max().item())

    print("\nHR information")
    print("-" * 30)

    print("Shape:", hr.shape)
    print("Data type:", hr.dtype)
    print("Minimum:", hr.min().item())
    print("Maximum:", hr.max().item())

    # --------------------------------------------------------
    # DataLoader test
    # --------------------------------------------------------

    print("\nTesting DataLoader...")

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    lr_batch, hr_batch = next(
        iter(loader)
    )

    print("\nDataLoader test successful!")

    print(
        "LR batch shape:",
        lr_batch.shape
    )

    print(
        "HR batch shape:",
        hr_batch.shape
    )

    print("\n" + "=" * 60)
    print("DATASET TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)