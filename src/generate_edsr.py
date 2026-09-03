import os
import torch
import matplotlib.pyplot as plt

from dataset_loader import SEN2NAIPDataset
from data_preprocess import normalize_image
from edsr_model import EDSR


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/edsr_test.pth"

OUTPUT_PATH = "results/edsr_comparison.png"

# Use a sample that we already know can be read
SAMPLE_INDEX = 4000


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("EDSR SUPER-RESOLUTION INFERENCE")
print("=" * 60)

print("\nUsing device:", DEVICE)

print("\nLoading EDSR model...")

model = EDSR(
    in_channels=4,
    out_channels=4,
    num_features=64,
    num_blocks=16,
    scale=4
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Load trained weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("EDSR model loaded successfully!")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading SEN2NAIPv2...")

dataset = SEN2NAIPDataset()

print(
    "Dataset loaded successfully!"
)

print(
    "Using sample:",
    SAMPLE_INDEX
)


# ============================================================
# LOAD LR + HR
# ============================================================

print("\nReading sample...")

lr, hr = dataset[SAMPLE_INDEX]

print(
    "LR shape:",
    lr.shape
)

print(
    "HR shape:",
    hr.shape
)


# ============================================================
# NORMALIZATION
# ============================================================

lr_normalized = normalize_image(
    lr
)

hr_normalized = normalize_image(
    hr
)


# Add batch dimension
lr_input = lr_normalized.unsqueeze(0)

lr_input = lr_input.to(DEVICE)


# ============================================================
# GENERATE EDSR OUTPUT
# ============================================================

print("\nGenerating super-resolved image...")

with torch.no_grad():

    sr = model(
        lr_input
    )


# Remove batch dimension
sr = sr.squeeze(0)

# Keep values between 0 and 1
sr = torch.clamp(
    sr,
    0.0,
    1.0
)

print(
    "SR shape:",
    sr.shape
)


# ============================================================
# BICUBIC UPSAMPLING
# ============================================================

print("\nGenerating Bicubic baseline...")

lr_batch = lr_normalized.unsqueeze(0)

bicubic = torch.nn.functional.interpolate(
    lr_batch,
    size=(
        hr_normalized.shape[1],
        hr_normalized.shape[2]
    ),
    mode="bicubic",
    align_corners=False
)

bicubic = bicubic.squeeze(0)

bicubic = torch.clamp(
    bicubic,
    0.0,
    1.0
)

print(
    "Bicubic shape:",
    bicubic.shape
)


# ============================================================
# CONVERT TO RGB
# ============================================================
#
# Dataset band order:
#
# B02 = Blue
# B03 = Green
# B04 = Red
# B08 = NIR
#
# RGB therefore:
#
# Red   = B04 → channel 2
# Green = B03 → channel 1
# Blue  = B02 → channel 0
#
# ============================================================

def create_rgb(image):

    red = image[2]
    green = image[1]
    blue = image[0]

    rgb = torch.stack(
        [
            red,
            green,
            blue
        ],
        dim=2
    )

    return rgb.cpu().numpy()


lr_rgb = create_rgb(
    lr_normalized
)

bicubic_rgb = create_rgb(
    bicubic
)

sr_rgb = create_rgb(
    sr
)

hr_rgb = create_rgb(
    hr_normalized
)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# VISUALIZATION
# ============================================================

print("\nCreating comparison figure...")

plt.figure(
    figsize=(16, 4)
)


# ------------------------------------------------------------
# LR
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    1
)

plt.imshow(
    lr_rgb
)

plt.title(
    "Input LR\n10 m"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# BICUBIC
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    2
)

plt.imshow(
    bicubic_rgb
)

plt.title(
    "Bicubic\nBaseline"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# EDSR
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    3
)

plt.imshow(
    sr_rgb
)

plt.title(
    "EDSR\nSuper-Resolved"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# HR REFERENCE
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    4
)

plt.imshow(
    hr_rgb
)

plt.title(
    "HR Reference\n2.5 m"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("EDSR VISUALIZATION COMPLETED!")
print("=" * 60)

print(
    "\nResult saved at:"
)

print(
    OUTPUT_PATH
)

print("\nPipeline:")

print(
    "LR → Bicubic → EDSR → HR Reference"
)

print(
    "\nNext task:"
)

print(
    "Calculate PSNR and SSIM for Bicubic vs EDSR."
)