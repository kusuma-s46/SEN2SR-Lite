import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio

from edsr_model import EDSR
from swinir_model import SwinIR


# ============================================================
# CONFIGURATION
# ============================================================

TEST_LR_DIR = "dataset/local/test/lr"
TEST_HR_DIR = "dataset/local/test/hr"

EDSR_MODEL_PATH = "models/edsr_local.pth"
SWINIR_MODEL_PATH = "models/swinir_local.pth"

OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sr_visual_comparison.png")

# Select which test image to visualize
# 0 = first test image
# 1 = second test image
# etc.
SAMPLE_INDEX = 0

# Sentinel-2 bands used by the project:
# Channel 0 = B02 Blue
# Channel 1 = B03 Green
# Channel 2 = B04 Red
# Channel 3 = B08 NIR

SCALE = 5000.0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("SUPER-RESOLUTION VISUAL COMPARISON")
print("=" * 70)

print(f"\nDevice: {DEVICE}")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTION: LOAD TIFF
# ============================================================

def load_tif(path):
    """
    Load a multispectral GeoTIFF.

    Returns:
        numpy array with shape [channels, height, width]
    """

    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)

    return image


# ============================================================
# HELPER FUNCTION: NORMALIZE
# ============================================================

def normalize(image):
    """
    Normalize Sentinel-2 pixel values to approximately 0-1.
    """

    image = image / SCALE
    image = np.clip(image, 0.0, 1.0)

    return image


# ============================================================
# HELPER FUNCTION: RGB CONVERSION
# ============================================================

def create_rgb(image):
    """
    Convert 4-band image to RGB.

    Input channels:
        0 = B02 Blue
        1 = B03 Green
        2 = B04 Red
        3 = B08 NIR

    RGB output:
        R = B04
        G = B03
        B = B02
    """

    red = image[2]
    green = image[1]
    blue = image[0]

    rgb = np.stack(
        [red, green, blue],
        axis=-1
    )

    return np.clip(rgb, 0.0, 1.0)


# ============================================================
# HELPER FUNCTION: IMPROVE DISPLAY
# ============================================================

def enhance_rgb(rgb):
    """
    Improve visualization contrast using percentile stretching.

    This is ONLY for visualization.
    It does not change the actual SR values.
    """

    output = np.zeros_like(rgb)

    for channel in range(3):

        band = rgb[:, :, channel]

        low = np.percentile(band, 2)
        high = np.percentile(band, 98)

        if high > low:
            band = (band - low) / (high - low)

        output[:, :, channel] = band

    return np.clip(output, 0.0, 1.0)


# ============================================================
# FIND TEST FILES
# ============================================================

lr_files = sorted(
    glob.glob(os.path.join(TEST_LR_DIR, "*.tif"))
)

hr_files = sorted(
    glob.glob(os.path.join(TEST_HR_DIR, "*.tif"))
)

print(f"\nLR test images found: {len(lr_files)}")
print(f"HR test images found: {len(hr_files)}")


# ============================================================
# CHECK DATASET
# ============================================================

if len(lr_files) == 0:
    raise FileNotFoundError(
        f"No LR TIFF files found in: {TEST_LR_DIR}"
    )

if len(hr_files) == 0:
    raise FileNotFoundError(
        f"No HR TIFF files found in: {TEST_HR_DIR}"
    )

if SAMPLE_INDEX >= len(lr_files):
    raise IndexError(
        f"SAMPLE_INDEX={SAMPLE_INDEX} is larger than available "
        f"test samples ({len(lr_files)})."
    )


# ============================================================
# SELECT SAMPLE
# ============================================================

lr_path = lr_files[SAMPLE_INDEX]
hr_path = hr_files[SAMPLE_INDEX]

print("\nSelected sample:")
print(f"LR: {lr_path}")
print(f"HR: {hr_path}")


# ============================================================
# LOAD IMAGE
# ============================================================

lr_image = load_tif(lr_path)
hr_image = load_tif(hr_path)

print("\nOriginal image shapes:")
print(f"LR shape: {lr_image.shape}")
print(f"HR shape: {hr_image.shape}")


# ============================================================
# NORMALIZE
# ============================================================

lr_norm = normalize(lr_image)
hr_norm = normalize(hr_image)


# ============================================================
# CONVERT TO PYTORCH TENSOR
# ============================================================

lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0).float()

lr_tensor = lr_tensor.to(DEVICE)

print(f"LR tensor shape: {lr_tensor.shape}")


# ============================================================
# BICUBIC UPSAMPLING
# ============================================================

print("\nGenerating Bicubic result...")

with torch.no_grad():

    bicubic_tensor = torch.nn.functional.interpolate(
        lr_tensor,
        size=(520, 520),
        mode="bicubic",
        align_corners=False
    )

print(f"Bicubic shape: {bicubic_tensor.shape}")


# ============================================================
# LOAD EDSR
# ============================================================

print("\nLoading EDSR model...")

edsr = EDSR(
    in_channels=4,
    out_channels=4,
    num_features=64,
    num_blocks=16,
    scale=4
)

if not os.path.exists(EDSR_MODEL_PATH):
    raise FileNotFoundError(
        f"EDSR model not found:\n{EDSR_MODEL_PATH}"
    )

edsr.load_state_dict(
    torch.load(
        EDSR_MODEL_PATH,
        map_location=DEVICE
    )
)

edsr = edsr.to(DEVICE)
edsr.eval()

print("EDSR loaded successfully.")


# ============================================================
# EDSR INFERENCE
# ============================================================

print("\nGenerating EDSR result...")

with torch.no_grad():

    edsr_tensor = edsr(lr_tensor)

print(f"EDSR shape: {edsr_tensor.shape}")


# ============================================================
# LOAD SWINIR
# ============================================================

print("\nLoading SwinIR model...")

swinir = SwinIR(
    in_channels=4,
    out_channels=4,
    embed_dim=64,
    depths=4,
    num_heads=4,
    window_size=8,
    upscale=4
)

if not os.path.exists(SWINIR_MODEL_PATH):
    raise FileNotFoundError(
        f"SwinIR model not found:\n{SWINIR_MODEL_PATH}"
    )

swinir.load_state_dict(
    torch.load(
        SWINIR_MODEL_PATH,
        map_location=DEVICE
    )
)

swinir = swinir.to(DEVICE)
swinir.eval()

print("SwinIR loaded successfully.")


# ============================================================
# SWINIR INFERENCE
# ============================================================

print("\nGenerating SwinIR result...")

with torch.no_grad():

    swinir_tensor = swinir(lr_tensor)

print(f"SwinIR shape: {swinir_tensor.shape}")


# ============================================================
# CONVERT TENSORS TO NUMPY
# ============================================================

bicubic = bicubic_tensor.squeeze(0).cpu().numpy()

edsr_sr = edsr_tensor.squeeze(0).cpu().numpy()

swinir_sr = swinir_tensor.squeeze(0).cpu().numpy()


# ============================================================
# CLIP MODEL OUTPUTS
# ============================================================

bicubic = np.clip(bicubic, 0.0, 1.0)

edsr_sr = np.clip(edsr_sr, 0.0, 1.0)

swinir_sr = np.clip(swinir_sr, 0.0, 1.0)


# ============================================================
# CREATE RGB IMAGES
# ============================================================

lr_rgb = create_rgb(lr_norm)

bicubic_rgb = create_rgb(bicubic)

edsr_rgb = create_rgb(edsr_sr)

swinir_rgb = create_rgb(swinir_sr)

hr_rgb = create_rgb(hr_norm)


# ============================================================
# ENHANCE ONLY FOR VISUALIZATION
# ============================================================

lr_display = enhance_rgb(lr_rgb)

bicubic_display = enhance_rgb(bicubic_rgb)

edsr_display = enhance_rgb(edsr_rgb)

swinir_display = enhance_rgb(swinir_rgb)

hr_display = enhance_rgb(hr_rgb)


# ============================================================
# PRINT IMAGE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("IMAGE INFORMATION")
print("=" * 70)

print(f"\nLR image:")
print(f"  Shape: {lr_image.shape}")
print(f"  Resolution: 130 × 130")

print(f"\nBicubic:")
print(f"  Shape: {bicubic.shape}")
print(f"  Resolution: 520 × 520")

print(f"\nEDSR:")
print(f"  Shape: {edsr_sr.shape}")
print(f"  Resolution: 520 × 520")

print(f"\nSwinIR:")
print(f"  Shape: {swinir_sr.shape}")
print(f"  Resolution: 520 × 520")

print(f"\nHR Reference:")
print(f"  Shape: {hr_image.shape}")
print(f"  Resolution: 520 × 520")


# ============================================================
# CREATE COMPARISON FIGURE
# ============================================================

print("\nCreating visualization...")

fig, axes = plt.subplots(
    1,
    5,
    figsize=(25, 6)
)


# ============================================================
# LR
# ============================================================

axes[0].imshow(lr_display)

axes[0].set_title(
    "LR Sentinel-2\n130 × 130",
    fontsize=14
)

axes[0].axis("off")


# ============================================================
# BICUBIC
# ============================================================

axes[1].imshow(bicubic_display)

axes[1].set_title(
    "Bicubic\n520 × 520",
    fontsize=14
)

axes[1].axis("off")


# ============================================================
# EDSR
# ============================================================

axes[2].imshow(edsr_display)

axes[2].set_title(
    "EDSR\n520 × 520",
    fontsize=14
)

axes[2].axis("off")


# ============================================================
# SWINIR
# ============================================================

axes[3].imshow(swinir_display)

axes[3].set_title(
    "SwinIR\n520 × 520",
    fontsize=14
)

axes[3].axis("off")


# ============================================================
# HR REFERENCE
# ============================================================

axes[4].imshow(hr_display)

axes[4].set_title(
    "HR Reference\n520 × 520",
    fontsize=14
)

axes[4].axis("off")


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Sentinel-2 Super-Resolution Visual Comparison",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# SAVE FIGURE
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

print("\nVisualization saved successfully!")

print(f"\nOutput file:")
print(OUTPUT_FILE)


# ============================================================
# DISPLAY
# ============================================================

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)

print("\nComparison:")
print("LR Sentinel-2 → Bicubic → EDSR → SwinIR → HR Reference")

print("\nThe generated image can be used in:")
print("1. Task 2 implementation")
print("2. Project report")
print("3. PPT presentation")
print("4. Model comparison")
print("5. SIH demonstration")