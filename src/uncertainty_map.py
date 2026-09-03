import os
import glob
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from edsr_model import EDSR


# ============================================================
# SETTINGS
# ============================================================

TEST_LR_DIR = "dataset/local/test/lr"
MODEL_PATH = "models/edsr_local.pth"

OUTPUT_DIR = "results/uncertainty"

MAX_SAMPLES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TIFF
# ============================================================

def load_tiff(path):

    with rasterio.open(path) as src:

        image = src.read().astype(
            np.float32
        )

    return image


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(image):

    return image / 5000.0


# ============================================================
# LOAD EDSR MODEL
# ============================================================

print("=" * 70)
print("EDSR UNCERTAINTY MAP GENERATION")
print("=" * 70)

print()
print("Device:", DEVICE)

print()
print("Loading EDSR model...")


model = EDSR(
    in_channels=4,
    out_channels=4,
    num_features=64,
    num_blocks=16,
    scale=4
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()


print("EDSR model loaded successfully.")


# ============================================================
# GET TEST IMAGES
# ============================================================

lr_files = sorted(
    glob.glob(
        os.path.join(
            TEST_LR_DIR,
            "*.tif"
        )
    )
)

lr_files = lr_files[:MAX_SAMPLES]


print()
print("Test images:", len(lr_files))


# ============================================================
# UNCERTAINTY METHOD
# ============================================================
#
# We generate several predictions by adding very small
# perturbations to the input image.
#
# The standard deviation between predictions is used as
# an uncertainty estimate.
#
# Higher variation  -> higher uncertainty
# Lower variation   -> higher confidence
#
# This is a practical uncertainty proxy for our project.
#


NUM_PREDICTIONS = 5

NOISE_LEVEL = 0.005


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for index, image_path in enumerate(
    lr_files
):

    print()
    print(
        f"Processing image "
        f"{index + 1}/{len(lr_files)}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    image = load_tiff(
        image_path
    )

    print(
        "Input shape:",
        image.shape
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image = normalize(
        image
    )

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(DEVICE)

    predictions = []


    # ========================================================
    # MULTIPLE PREDICTIONS
    # ========================================================

    for p in range(
        NUM_PREDICTIONS
    ):

        # Generate small random perturbation

        noise = torch.randn_like(
            image_tensor
        ) * NOISE_LEVEL

        perturbed_input = (
            image_tensor + noise
        )

        perturbed_input = torch.clamp(
            perturbed_input,
            0,
            1
        )


        # ----------------------------------------------------
        # EDSR prediction
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                perturbed_input
            )


        output = torch.clamp(
            output,
            0,
            1
        )


        output = (
            output
            .squeeze(0)
            .cpu()
            .numpy()
        )


        predictions.append(
            output
        )


    # ========================================================
    # CONVERT PREDICTIONS
    # ========================================================

    predictions = np.array(
        predictions
    )

    print(
        "Prediction shape:",
        predictions.shape
    )


    # ========================================================
    # MEAN SR IMAGE
    # ========================================================

    mean_sr = np.mean(
        predictions,
        axis=0
    )


    # ========================================================
    # UNCERTAINTY
    # ========================================================
    #
    # Pixel-wise standard deviation
    #

    uncertainty = np.std(
        predictions,
        axis=0
    )


    # Average uncertainty across
    # the four spectral bands

    uncertainty_map = np.mean(
        uncertainty,
        axis=0
    )


    # ========================================================
    # NORMALIZE UNCERTAINTY MAP
    # ========================================================

    min_value = np.min(
        uncertainty_map
    )

    max_value = np.max(
        uncertainty_map
    )


    if max_value > min_value:

        uncertainty_normalized = (
            uncertainty_map - min_value
        ) / (
            max_value - min_value
        )

    else:

        uncertainty_normalized = (
            np.zeros_like(
                uncertainty_map
            )
        )


    # ========================================================
    # CONFIDENCE MAP
    # ========================================================

    confidence_map = (
        1.0 - uncertainty_normalized
    )


    # ========================================================
    # FILE NAME
    # ========================================================

    filename = os.path.basename(
        image_path
    )

    filename = os.path.splitext(
        filename
    )[0]


    # ========================================================
    # SAVE NUMPY ARRAYS
    # ========================================================

    np.save(
        os.path.join(
            OUTPUT_DIR,
            filename + "_sr.npy"
        ),
        mean_sr
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            filename + "_uncertainty.npy"
        ),
        uncertainty_normalized
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            filename + "_confidence.npy"
        ),
        confidence_map
    )


    # ========================================================
    # RGB VISUALIZATION
    # ========================================================

    # B04 = Red
    # B03 = Green
    # B02 = Blue

    red = mean_sr[2]
    green = mean_sr[1]
    blue = mean_sr[0]


    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )


    # Normalize RGB for display

    rgb_min = np.percentile(
        rgb,
        2
    )

    rgb_max = np.percentile(
        rgb,
        98
    )


    rgb = np.clip(
        (rgb - rgb_min)
        / (rgb_max - rgb_min + 1e-8),
        0,
        1
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )


    # --------------------------------------------------------
    # SR IMAGE
    # --------------------------------------------------------

    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        rgb
    )

    plt.title(
        "EDSR Super-Resolved Image"
    )

    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        uncertainty_normalized,
        cmap="inferno"
    )

    plt.colorbar(
        fraction=0.046,
        pad=0.04
    )

    plt.title(
        "Uncertainty Map"
    )

    plt.axis(
        "off"
    )


    plt.tight_layout()


    # ========================================================
    # SAVE FIGURE
    # ========================================================

    output_image = os.path.join(
        OUTPUT_DIR,
        filename + "_uncertainty.png"
    )


    plt.savefig(
        output_image,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # PRINT STATISTICS
    # ========================================================

    print(
        "Uncertainty minimum:",
        f"{uncertainty_map.min():.6f}"
    )

    print(
        "Uncertainty maximum:",
        f"{uncertainty_map.max():.6f}"
    )

    print(
        "Uncertainty mean:",
        f"{uncertainty_map.mean():.6f}"
    )

    print(
        "Saved:",
        output_image
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("UNCERTAINTY MAP GENERATION COMPLETE")
print("=" * 70)

print()
print(
    "Results saved in:",
    OUTPUT_DIR
)