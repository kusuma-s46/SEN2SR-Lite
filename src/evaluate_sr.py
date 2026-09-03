import os
import numpy as np
import torch
import torch.nn.functional as F
import rasterio

from skimage.metrics import structural_similarity as ssim

from edsr_model import EDSR


# ============================================================
# CONFIGURATION
# ============================================================

TEST_LR_DIR = "dataset/local/test/lr"
TEST_HR_DIR = "dataset/local/test/hr"

MODEL_PATH = "models/edsr_local.pth"

RESULTS_DIR = "results"

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "edsr_local_evaluation.txt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# READ TIFF IMAGE
# ============================================================

def read_tiff(path):

    with rasterio.open(path) as src:
        image = src.read().astype("float32")

    return torch.from_numpy(image)


# ============================================================
# NORMALIZE IMAGE
# ============================================================

def normalize_image(image):

    return image / 5000.0


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(predicted, target):

    mse = torch.mean(
        (predicted - target) ** 2
    ).item()

    if mse == 0:
        return 100.0

    return 10 * np.log10(1.0 / mse)


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(predicted, target):

    predicted = predicted.cpu().numpy()
    target = target.cpu().numpy()

    band_scores = []

    # Calculate SSIM separately for each spectral band
    for band in range(predicted.shape[0]):

        score = ssim(
            target[band],
            predicted[band],
            data_range=1.0
        )

        band_scores.append(score)

    return np.mean(band_scores)


# ============================================================
# BICUBIC UPSAMPLING
# ============================================================

def generate_bicubic(lr):

    bicubic = F.interpolate(
        lr.unsqueeze(0),
        size=(520, 520),
        mode="bicubic",
        align_corners=False
    )

    return bicubic.squeeze(0)


# ============================================================
# LOAD EDSR MODEL
# ============================================================

def load_model():

    print("\nLoading EDSR model...")

    model = EDSR(
        in_channels=4,
        out_channels=4,
        num_features=64,
        num_blocks=16,
        scale=4
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.eval()

    print("EDSR model loaded successfully!")

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LOCAL EDSR SUPER-RESOLUTION EVALUATION")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not os.path.exists(TEST_LR_DIR):

        print("\nERROR: Test LR directory not found!")
        print(TEST_LR_DIR)
        return

    if not os.path.exists(TEST_HR_DIR):

        print("\nERROR: Test HR directory not found!")
        print(TEST_HR_DIR)
        return

    if not os.path.exists(MODEL_PATH):

        print("\nERROR: EDSR model not found!")
        print(MODEL_PATH)
        return

    # --------------------------------------------------------
    # Get test files
    # --------------------------------------------------------

    test_files = sorted([
        f for f in os.listdir(TEST_LR_DIR)
        if f.lower().endswith(".tif")
    ])

    if len(test_files) == 0:

        print("\nERROR: No test images found!")

        return

    print("\nLocal test dataset loaded.")

    print(
        "Number of test pairs:",
        len(test_files)
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Metric storage
    # --------------------------------------------------------

    bicubic_psnr_scores = []
    bicubic_ssim_scores = []

    edsr_psnr_scores = []
    edsr_ssim_scores = []

    # --------------------------------------------------------
    # Evaluate every test image
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("EVALUATING TEST IMAGES")
    print("=" * 60)

    for index, filename in enumerate(test_files):

        print(
            f"\nTest image {index + 1}/{len(test_files)}"
        )

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        lr_path = os.path.join(
            TEST_LR_DIR,
            filename
        )

        hr_path = os.path.join(
            TEST_HR_DIR,
            filename
        )

        # ----------------------------------------------------
        # Check matching HR image
        # ----------------------------------------------------

        if not os.path.exists(hr_path):

            print(
                "WARNING: HR image missing:",
                filename
            )

            continue

        # ----------------------------------------------------
        # Read images
        # ----------------------------------------------------

        lr = read_tiff(lr_path)
        hr = read_tiff(hr_path)

        print(
            "LR shape:",
            tuple(lr.shape)
        )

        print(
            "HR shape:",
            tuple(hr.shape)
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        lr = normalize_image(lr)
        hr = normalize_image(hr)

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        lr_input = lr.unsqueeze(0).to(DEVICE)
        hr = hr.to(DEVICE)

        # ----------------------------------------------------
        # Bicubic
        # ----------------------------------------------------

        bicubic = generate_bicubic(
            lr.to(DEVICE)
        )

        bicubic = torch.clamp(
            bicubic,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # EDSR
        # ----------------------------------------------------

        with torch.no_grad():

            edsr = model(
                lr_input
            ).squeeze(0)

        edsr = torch.clamp(
            edsr,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Calculate Bicubic metrics
        # ----------------------------------------------------

        bicubic_psnr = calculate_psnr(
            bicubic,
            hr
        )

        bicubic_ssim = calculate_ssim(
            bicubic,
            hr
        )

        # ----------------------------------------------------
        # Calculate EDSR metrics
        # ----------------------------------------------------

        edsr_psnr = calculate_psnr(
            edsr,
            hr
        )

        edsr_ssim = calculate_ssim(
            edsr,
            hr
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        bicubic_psnr_scores.append(
            bicubic_psnr
        )

        bicubic_ssim_scores.append(
            bicubic_ssim
        )

        edsr_psnr_scores.append(
            edsr_psnr
        )

        edsr_ssim_scores.append(
            edsr_ssim
        )

        # ----------------------------------------------------
        # Print current result
        # ----------------------------------------------------

        print(
            f"Bicubic PSNR : {bicubic_psnr:.4f} dB"
        )

        print(
            f"Bicubic SSIM : {bicubic_ssim:.4f}"
        )

        print(
            f"EDSR PSNR    : {edsr_psnr:.4f} dB"
        )

        print(
            f"EDSR SSIM    : {edsr_ssim:.4f}"
        )

    # ========================================================
    # FINAL AVERAGE RESULTS
    # ========================================================

    if len(edsr_psnr_scores) == 0:

        print("\nNo valid test samples evaluated.")

        return

    avg_bicubic_psnr = np.mean(
        bicubic_psnr_scores
    )

    avg_bicubic_ssim = np.mean(
        bicubic_ssim_scores
    )

    avg_edsr_psnr = np.mean(
        edsr_psnr_scores
    )

    avg_edsr_ssim = np.mean(
        edsr_ssim_scores
    )

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    psnr_improvement = (
        avg_edsr_psnr -
        avg_bicubic_psnr
    )

    ssim_improvement = (
        avg_edsr_ssim -
        avg_bicubic_ssim
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL SUPER-RESOLUTION EVALUATION RESULTS")
    print("=" * 60)

    print("\nNumber of test images evaluated:")
    print(len(edsr_psnr_scores))

    print("\n")
    print(
        f"{'Method':<20}"
        f"{'PSNR (dB)':<20}"
        f"{'SSIM':<15}"
    )

    print("-" * 55)

    print(
        f"{'Bicubic':<20}"
        f"{avg_bicubic_psnr:<20.4f}"
        f"{avg_bicubic_ssim:<15.4f}"
    )

    print(
        f"{'EDSR':<20}"
        f"{avg_edsr_psnr:<20.4f}"
        f"{avg_edsr_ssim:<15.4f}"
    )

    print("-" * 55)

    print("\n")
    print("IMPROVEMENT OF EDSR OVER BICUBIC")
    print("-" * 55)

    print(
        f"PSNR improvement: "
        f"{psnr_improvement:+.4f} dB"
    )

    print(
        f"SSIM improvement: "
        f"{ssim_improvement:+.4f}"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print("\n")
    print("INTERPRETATION")
    print("-" * 55)

    if psnr_improvement > 0:

        print(
            "EDSR exceeds Bicubic in average PSNR."
        )

    else:

        print(
            "EDSR does not yet exceed Bicubic "
            "in average PSNR."
        )

    if ssim_improvement > 0:

        print(
            "EDSR exceeds Bicubic in average SSIM."
        )

    else:

        print(
            "EDSR does not yet exceed Bicubic "
            "in average SSIM."
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w"
    ) as file:

        file.write(
            "LOCAL EDSR SUPER-RESOLUTION EVALUATION\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Number of test images: "
            f"{len(edsr_psnr_scores)}\n\n"
        )

        file.write(
            "MODEL PERFORMANCE\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            f"Bicubic PSNR: "
            f"{avg_bicubic_psnr:.4f} dB\n"
        )

        file.write(
            f"Bicubic SSIM: "
            f"{avg_bicubic_ssim:.4f}\n\n"
        )

        file.write(
            f"EDSR PSNR: "
            f"{avg_edsr_psnr:.4f} dB\n"
        )

        file.write(
            f"EDSR SSIM: "
            f"{avg_edsr_ssim:.4f}\n\n"
        )

        file.write(
            "IMPROVEMENT\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            f"PSNR improvement: "
            f"{psnr_improvement:+.4f} dB\n"
        )

        file.write(
            f"SSIM improvement: "
            f"{ssim_improvement:+.4f}\n"
        )

    print("\n")
    print("Results saved to:")

    print(OUTPUT_FILE)

    print("\n")
    print("=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()