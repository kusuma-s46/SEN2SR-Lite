import os
import numpy as np
import torch
import rasterio
from skimage.metrics import structural_similarity as ssim

from edsr_model import EDSR


# ============================================================
# CONFIGURATION
# ============================================================

TEST_LR_DIR = "dataset/local/test/lr"
TEST_HR_DIR = "dataset/local/test/hr"

MODEL_PATH = "models/edsr_local.pth"

OUTPUT_DIR = "results/high_resolution_validation"

MAX_SAMPLES = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SCALE = 5000.0

BANDS = ["B02", "B03", "B04", "B08"]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PSNR FUNCTION
# ============================================================

def calculate_psnr(reference, prediction):

    mse = np.mean(
        (reference - prediction) ** 2
    )

    if mse == 0:
        return 100.0

    return 10 * np.log10(
        1.0 / mse
    )


# ============================================================
# SSIM FUNCTION
# ============================================================

def calculate_ssim(reference, prediction):

    return ssim(
        reference,
        prediction,
        data_range=1.0
    )


# ============================================================
# LOAD EDSR
# ============================================================

print("=" * 70)
print("HIGH-RESOLUTION REFERENCE VALIDATION")
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
# FIND TEST PAIRS
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

for filename in lr_files:

    if filename in hr_files:
        pairs.append(filename)

pairs = pairs[:MAX_SAMPLES]

print("\nNumber of validation samples:", len(pairs))


# ============================================================
# RESULT STORAGE
# ============================================================

psnr_results = {
    "B02": [],
    "B03": [],
    "B04": [],
    "B08": []
}

ssim_results = {
    "B02": [],
    "B03": [],
    "B04": [],
    "B08": []
}

overall_psnr = []
overall_ssim = []


# ============================================================
# PROCESS TEST IMAGES
# ============================================================

for index, filename in enumerate(pairs):

    print(
        f"\nProcessing {index + 1}/{len(pairs)}: {filename}"
    )

    lr_path = os.path.join(
        TEST_LR_DIR,
        filename
    )

    hr_path = os.path.join(
        TEST_HR_DIR,
        filename
    )

    # --------------------------------------------------------
    # READ LR
    # --------------------------------------------------------

    with rasterio.open(lr_path) as src:

        lr = src.read().astype(
            np.float32
        )

    # --------------------------------------------------------
    # READ HR
    # --------------------------------------------------------

    with rasterio.open(hr_path) as src:

        hr = src.read().astype(
            np.float32
        )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    lr_normalized = np.clip(
        lr / SCALE,
        0,
        1
    )

    hr_normalized = np.clip(
        hr / SCALE,
        0,
        1
    )

    # --------------------------------------------------------
    # LR → EDSR → SR
    # --------------------------------------------------------

    lr_tensor = torch.from_numpy(
        lr_normalized
    ).unsqueeze(0).float().to(DEVICE)

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

    # --------------------------------------------------------
    # CHECK SHAPES
    # --------------------------------------------------------

    if sr.shape != hr_normalized.shape:

        print(
            "Shape mismatch:"
        )

        print(
            "SR:",
            sr.shape
        )

        print(
            "HR:",
            hr_normalized.shape
        )

        continue

    # --------------------------------------------------------
    # BAND-WISE VALIDATION
    # --------------------------------------------------------

    sample_psnr = []
    sample_ssim = []

    for band_index, band_name in enumerate(BANDS):

        reference = hr_normalized[
            band_index
        ]

        prediction = sr[
            band_index
        ]

        # PSNR
        band_psnr = calculate_psnr(
            reference,
            prediction
        )

        # SSIM
        band_ssim = calculate_ssim(
            reference,
            prediction
        )

        psnr_results[
            band_name
        ].append(
            band_psnr
        )

        ssim_results[
            band_name
        ].append(
            band_ssim
        )

        sample_psnr.append(
            band_psnr
        )

        sample_ssim.append(
            band_ssim
        )

        print(
            f"  {band_name}: "
            f"PSNR={band_psnr:.4f} dB, "
            f"SSIM={band_ssim:.4f}"
        )

    # --------------------------------------------------------
    # IMAGE AVERAGE
    # --------------------------------------------------------

    image_psnr = np.mean(
        sample_psnr
    )

    image_ssim = np.mean(
        sample_ssim
    )

    overall_psnr.append(
        image_psnr
    )

    overall_ssim.append(
        image_ssim
    )

    print(
        f"  Image Average: "
        f"PSNR={image_psnr:.4f} dB, "
        f"SSIM={image_ssim:.4f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL HIGH-RESOLUTION VALIDATION RESULTS")
print("=" * 70)


final_band_results = {}


for band in BANDS:

    mean_psnr = np.mean(
        psnr_results[band]
    )

    mean_ssim = np.mean(
        ssim_results[band]
    )

    final_band_results[
        band
    ] = {
        "psnr": mean_psnr,
        "ssim": mean_ssim
    }

    print(
        f"\n{band}"
    )

    print(
        f"  Average PSNR : "
        f"{mean_psnr:.4f} dB"
    )

    print(
        f"  Average SSIM : "
        f"{mean_ssim:.4f}"
    )


# ============================================================
# OVERALL RESULT
# ============================================================

average_psnr = np.mean(
    overall_psnr
)

average_ssim = np.mean(
    overall_ssim
)


print("\n")
print("=" * 70)
print("OVERALL VALIDATION")
print("=" * 70)

print(
    f"\nAverage PSNR : "
    f"{average_psnr:.4f} dB"
)

print(
    f"Average SSIM : "
    f"{average_ssim:.4f}"
)


# ============================================================
# QUALITY INTERPRETATION
# ============================================================

if average_psnr >= 30:

    quality = (
        "High reconstruction quality."
    )

elif average_psnr >= 25:

    quality = (
        "Good reconstruction quality."
    )

elif average_psnr >= 20:

    quality = (
        "Moderate reconstruction quality."
    )

else:

    quality = (
        "Low reconstruction quality. "
        "The current model requires further training."
    )


print(
    "\nInterpretation:"
)

print(
    quality
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "high_resolution_validation.txt"
)


with open(
    report_path,
    "w"
) as f:

    f.write(
        "HIGH-RESOLUTION REFERENCE VALIDATION\n"
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
        "Output Resolution: 2.5 m\n"
    )

    f.write(
        "Scale Factor: 4x\n"
    )

    f.write(
        "Bands: B02, B03, B04, B08\n"
    )

    f.write(
        f"Validation Samples: {len(pairs)}\n\n"
    )

    f.write(
        "BAND-WISE RESULTS\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    for band in BANDS:

        f.write(
            f"{band}: "
            f"PSNR = "
            f"{final_band_results[band]['psnr']:.4f} dB, "
            f"SSIM = "
            f"{final_band_results[band]['ssim']:.4f}\n"
        )

    f.write(
        "\nOVERALL RESULTS\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        f"Average PSNR: "
        f"{average_psnr:.4f} dB\n"
    )

    f.write(
        f"Average SSIM: "
        f"{average_ssim:.4f}\n"
    )

    f.write(
        f"\nInterpretation: "
        f"{quality}\n"
    )


# ============================================================
# COMPLETED
# ============================================================

print(
    f"\nReport saved to:"
)

print(
    report_path
)

print(
    "\nHigh-resolution validation completed successfully."
)