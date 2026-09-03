import os
import numpy as np
import torch
import rasterio

from edsr_model import EDSR


# ============================================================
# CONFIGURATION
# ============================================================

TEST_LR_DIR = "dataset/local/test/lr"
TEST_HR_DIR = "dataset/local/test/hr"

MODEL_PATH = "models/edsr_local.pth"

OUTPUT_DIR = "results/spectral_consistency"

MAX_SAMPLES = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sentinel-2 bands used in the project
BANDS = ["B02", "B03", "B04", "B08"]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD EDSR MODEL
# ============================================================

print("=" * 70)
print("SPECTRAL CONSISTENCY ANALYSIS")
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

model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

print("EDSR model loaded successfully.")


# ============================================================
# GET FILES
# ============================================================

lr_files = sorted([
    f for f in os.listdir(TEST_LR_DIR)
    if f.lower().endswith(".tif")
])

hr_files = sorted([
    f for f in os.listdir(TEST_HR_DIR)
    if f.lower().endswith(".tif")
])

pairs = []

for lr_file in lr_files:

    if lr_file in hr_files:
        pairs.append((lr_file, lr_file))

pairs = pairs[:MAX_SAMPLES]

print("\nTest samples:", len(pairs))


# ============================================================
# METRICS STORAGE
# ============================================================

all_band_errors = {
    "B02": [],
    "B03": [],
    "B04": [],
    "B08": []
}

all_band_correlations = {
    "B02": [],
    "B03": [],
    "B04": [],
    "B08": []
}


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for index, (lr_file, hr_file) in enumerate(pairs):

    print(
        f"\nProcessing {index + 1}/{len(pairs)}: {lr_file}"
    )

    # --------------------------------------------------------
    # Read LR image
    # --------------------------------------------------------

    with rasterio.open(
        os.path.join(TEST_LR_DIR, lr_file)
    ) as src:

        lr = src.read().astype(np.float32)

    # --------------------------------------------------------
    # Read HR reference
    # --------------------------------------------------------

    with rasterio.open(
        os.path.join(TEST_HR_DIR, hr_file)
    ) as src:

        hr = src.read().astype(np.float32)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    lr_norm = lr / 5000.0

    lr_tensor = torch.from_numpy(
        lr_norm
    ).unsqueeze(0).float().to(DEVICE)

    # --------------------------------------------------------
    # Generate SR image
    # --------------------------------------------------------

    with torch.no_grad():

        sr_tensor = model(
            lr_tensor
        )

    sr = sr_tensor.squeeze(0).cpu().numpy()

    # --------------------------------------------------------
    # Clip values
    # --------------------------------------------------------

    sr = np.clip(sr, 0, 1)

    # Convert back to original scale
    sr = sr * 5000.0

    # --------------------------------------------------------
    # Compare spectral information
    # --------------------------------------------------------

    for band_index, band_name in enumerate(BANDS):

        sr_band = sr[band_index]
        hr_band = hr[band_index]

        # ----------------------------------------------------
        # Mean Absolute Error
        # ----------------------------------------------------

        mae = np.mean(
            np.abs(sr_band - hr_band)
        )

        # Normalize MAE
        mae_normalized = mae / 5000.0

        all_band_errors[band_name].append(
            mae_normalized
        )

        # ----------------------------------------------------
        # Spectral correlation
        # ----------------------------------------------------

        sr_flat = sr_band.flatten()
        hr_flat = hr_band.flatten()

        correlation = np.corrcoef(
            sr_flat,
            hr_flat
        )[0, 1]

        if np.isnan(correlation):
            correlation = 0.0

        all_band_correlations[band_name].append(
            correlation
        )


# ============================================================
# CALCULATE FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SPECTRAL CONSISTENCY RESULTS")
print("=" * 70)

results = []

for band in BANDS:

    mean_error = np.mean(
        all_band_errors[band]
    )

    mean_correlation = np.mean(
        all_band_correlations[band]
    )

    print(f"\n{band}")
    print("-" * 30)

    print(
        f"Normalized MAE       : {mean_error:.6f}"
    )

    print(
        f"Spectral Correlation : {mean_correlation:.6f}"
    )

    results.append(
        (
            band,
            mean_error,
            mean_correlation
        )
    )


# ============================================================
# OVERALL SPECTRAL CONSISTENCY
# ============================================================

overall_error = np.mean([
    r[1] for r in results
])

overall_correlation = np.mean([
    r[2] for r in results
])

print("\n")
print("=" * 70)
print("OVERALL RESULT")
print("=" * 70)

print(
    f"\nAverage Normalized MAE       : {overall_error:.6f}"
)

print(
    f"Average Spectral Correlation : {overall_correlation:.6f}"
)


# ============================================================
# INTERPRETATION
# ============================================================

if overall_correlation >= 0.90:

    interpretation = (
        "Very strong spectral consistency. "
        "The SR output preserves spectral patterns well."
    )

elif overall_correlation >= 0.75:

    interpretation = (
        "Good spectral consistency. "
        "Most spectral information is preserved."
    )

elif overall_correlation >= 0.50:

    interpretation = (
        "Moderate spectral consistency. "
        "Some spectral information may be altered."
    )

else:

    interpretation = (
        "Low spectral consistency. "
        "The SR model may significantly alter spectral information."
    )

print("\nInterpretation:")
print(interpretation)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "spectral_consistency_results.txt"
)

with open(output_file, "w") as f:

    f.write("SPECTRAL CONSISTENCY ANALYSIS\n")
    f.write("=" * 60 + "\n\n")

    f.write("Model: EDSR\n")
    f.write("Dataset: SEN2NAIPv2\n")
    f.write("Input: 10 m Sentinel-2\n")
    f.write("Output: 2.5 m Super-Resolved Image\n")
    f.write("Bands: B02, B03, B04, B08\n")
    f.write(f"Test Samples: {len(pairs)}\n\n")

    f.write("Band-wise Results\n")
    f.write("-" * 60 + "\n")

    for band, error, correlation in results:

        f.write(
            f"{band}: "
            f"Normalized MAE = {error:.6f}, "
            f"Correlation = {correlation:.6f}\n"
        )

    f.write("\n")
    f.write(
        f"Average Normalized MAE: "
        f"{overall_error:.6f}\n"
    )

    f.write(
        f"Average Spectral Correlation: "
        f"{overall_correlation:.6f}\n"
    )

    f.write("\nInterpretation:\n")
    f.write(interpretation)


print(
    f"\nResults saved to:\n{output_file}"
)

print("\nSpectral consistency analysis completed.")