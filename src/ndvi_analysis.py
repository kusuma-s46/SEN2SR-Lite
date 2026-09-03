import os
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt

from edsr_model import EDSR


# ============================================================
# CONFIGURATION
# ============================================================

TEST_LR_DIR = "dataset/local/test/lr"

MODEL_PATH = "models/edsr_local.pth"

OUTPUT_DIR = "results/ndvi"

MAX_SAMPLES = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SCALE = 5000.0

# Sentinel-2 band positions in our dataset
# B02 = Blue
# B03 = Green
# B04 = Red
# B08 = NIR

BLUE_INDEX = 0
GREEN_INDEX = 1
RED_INDEX = 2
NIR_INDEX = 3


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD EDSR MODEL
# ============================================================

print("=" * 70)
print("NDVI ANALYSIS USING SUPER-RESOLVED SENTINEL-2 IMAGE")
print("=" * 70)

print("\nDevice:", DEVICE)

model = EDSR(
    in_channels=4,
    out_channels=4,
    num_features=64,
    num_blocks=16,
    scale=4
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint
)

model.to(DEVICE)
model.eval()

print("EDSR model loaded successfully.")


# ============================================================
# GET TEST IMAGES
# ============================================================

lr_files = sorted([
    f for f in os.listdir(TEST_LR_DIR)
    if f.lower().endswith(".tif")
])

lr_files = lr_files[:MAX_SAMPLES]

print(
    "\nNumber of images:",
    len(lr_files)
)


# ============================================================
# STORAGE
# ============================================================

all_mean_ndvi = []
all_min_ndvi = []
all_max_ndvi = []

saved_first_map = False


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for index, filename in enumerate(lr_files):

    print(
        f"\nProcessing {index + 1}/{len(lr_files)}: {filename}"
    )

    lr_path = os.path.join(
        TEST_LR_DIR,
        filename
    )

    # --------------------------------------------------------
    # READ LOW-RESOLUTION IMAGE
    # --------------------------------------------------------

    with rasterio.open(lr_path) as src:

        lr = src.read().astype(
            np.float32
        )

    print(
        "LR shape:",
        lr.shape
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    lr_normalized = np.clip(
        lr / SCALE,
        0,
        1
    )

    # --------------------------------------------------------
    # CONVERT TO PYTORCH TENSOR
    # --------------------------------------------------------

    lr_tensor = torch.from_numpy(
        lr_normalized
    ).unsqueeze(0).float().to(DEVICE)

    # --------------------------------------------------------
    # SUPER-RESOLUTION
    # --------------------------------------------------------

    with torch.no_grad():

        sr_tensor = model(
            lr_tensor
        )

    sr = sr_tensor.squeeze(
        0
    ).cpu().numpy()

    sr = np.clip(
        sr,
        0,
        1
    )

    print(
        "SR shape:",
        sr.shape
    )

    # ========================================================
    # EXTRACT RED AND NIR
    # ========================================================

    red = sr[RED_INDEX]

    nir = sr[NIR_INDEX]

    # ========================================================
    # NDVI CALCULATION
    # ========================================================

    denominator = nir + red

    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(nir),
        where=denominator != 0
    )

    # Limit theoretical NDVI range
    ndvi = np.clip(
        ndvi,
        -1,
        1
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    mean_ndvi = float(
        np.mean(ndvi)
    )

    min_ndvi = float(
        np.min(ndvi)
    )

    max_ndvi = float(
        np.max(ndvi)
    )

    median_ndvi = float(
        np.median(ndvi)
    )

    vegetation_percentage = (
        np.sum(ndvi > 0.3)
        /
        ndvi.size
    ) * 100

    low_vegetation_percentage = (
        np.sum(
            (ndvi >= 0.1) &
            (ndvi <= 0.3)
        )
        /
        ndvi.size
    ) * 100

    barren_percentage = (
        np.sum(ndvi < 0.1)
        /
        ndvi.size
    ) * 100

    print(
        f"Mean NDVI       : {mean_ndvi:.4f}"
    )

    print(
        f"Minimum NDVI    : {min_ndvi:.4f}"
    )

    print(
        f"Maximum NDVI    : {max_ndvi:.4f}"
    )

    print(
        f"Median NDVI     : {median_ndvi:.4f}"
    )

    print(
        f"Vegetation >0.3 : {vegetation_percentage:.2f}%"
    )

    # ========================================================
    # STORE STATISTICS
    # ========================================================

    all_mean_ndvi.append(
        mean_ndvi
    )

    all_min_ndvi.append(
        min_ndvi
    )

    all_max_ndvi.append(
        max_ndvi
    )

    # ========================================================
    # SAVE NDVI ARRAY
    # ========================================================

    base_name = os.path.splitext(
        filename
    )[0]

    np.save(
        os.path.join(
            OUTPUT_DIR,
            base_name + "_ndvi.npy"
        ),
        ndvi
    )

    # ========================================================
    # SAVE FIRST NDVI MAP
    # ========================================================

    if not saved_first_map:

        plt.figure(
            figsize=(8, 6)
        )

        plt.imshow(
            ndvi,
            cmap="RdYlGn",
            vmin=-1,
            vmax=1
        )

        plt.colorbar(
            label="NDVI"
        )

        plt.title(
            "NDVI Map from EDSR Super-Resolved Image"
        )

        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "ndvi_map.png"
            ),
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        saved_first_map = True


# ============================================================
# OVERALL STATISTICS
# ============================================================

average_ndvi = np.mean(
    all_mean_ndvi
)

average_min_ndvi = np.mean(
    all_min_ndvi
)

average_max_ndvi = np.mean(
    all_max_ndvi
)


# ============================================================
# NDVI INTERPRETATION
# ============================================================

if average_ndvi >= 0.6:

    interpretation = (
        "High vegetation density."
    )

elif average_ndvi >= 0.3:

    interpretation = (
        "Moderate to healthy vegetation."
    )

elif average_ndvi >= 0.1:

    interpretation = (
        "Low vegetation or sparse vegetation."
    )

else:

    interpretation = (
        "Very low vegetation or predominantly "
        "non-vegetated surface."
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL NDVI RESULTS")
print("=" * 70)

print(
    f"\nAverage Mean NDVI : "
    f"{average_ndvi:.4f}"
)

print(
    f"Average Minimum NDVI : "
    f"{average_min_ndvi:.4f}"
)

print(
    f"Average Maximum NDVI : "
    f"{average_max_ndvi:.4f}"
)

print(
    "\nInterpretation:"
)

print(
    interpretation
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "ndvi_report.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(
        "NDVI ANALYSIS REPORT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "Model: EDSR\n"
    )

    f.write(
        "Dataset: SEN2NAIPv2\n"
    )

    f.write(
        "Input Resolution: 10 m\n"
    )

    f.write(
        "Super-Resolved Resolution: 2.5 m\n"
    )

    f.write(
        "Scale Factor: 4x\n"
    )

    f.write(
        "Red Band: B04\n"
    )

    f.write(
        "NIR Band: B08\n"
    )

    f.write(
        f"Number of Images: {len(lr_files)}\n\n"
    )

    f.write(
        "NDVI Formula:\n"
    )

    f.write(
        "NDVI = (B08 - B04) / (B08 + B04)\n\n"
    )

    f.write(
        f"Average Mean NDVI: "
        f"{average_ndvi:.4f}\n"
    )

    f.write(
        f"Average Minimum NDVI: "
        f"{average_min_ndvi:.4f}\n"
    )

    f.write(
        f"Average Maximum NDVI: "
        f"{average_max_ndvi:.4f}\n"
    )

    f.write(
        "\nInterpretation:\n"
    )

    f.write(
        interpretation
    )


# ============================================================
# COMPLETED
# ============================================================

print(
    "\nNDVI map saved to:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "ndvi_map.png"
    )
)

print(
    "\nNDVI report saved to:"
)

print(
    report_path
)

print(
    "\nNDVI analysis completed successfully."
)