import os
import glob

import torch
import torch.nn.functional as F
import numpy as np
import rasterio

from skimage.metrics import structural_similarity

from edsr_model import EDSR
from swinir_model import SwinIR


# ============================================================
# Configuration
# ============================================================

TEST_LR_DIR = "dataset/local/test/lr"
TEST_HR_DIR = "dataset/local/test/hr"

EDSR_MODEL_PATH = "models/edsr_local.pth"
SWINIR_MODEL_PATH = "models/swinir_local.pth"

RESULT_FILE = "results/sr_model_comparison.txt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_TEST_SAMPLES = 20


# ============================================================
# Load GeoTIFF
# ============================================================

def load_tiff(path):

    with rasterio.open(path) as src:
        image = src.read()

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    return image


# ============================================================
# Normalize
# ============================================================

def normalize(image):

    return image / 5000.0


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(predicted, target):

    mse = torch.mean(
        (predicted - target) ** 2
    )

    if mse.item() == 0:
        return 100.0

    psnr = (
        10 *
        torch.log10(1.0 / mse)
    )

    return psnr.item()


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(predicted, target):

    predicted = predicted.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    scores = []

    for band in range(predicted.shape[0]):

        score = structural_similarity(
            target[band],
            predicted[band],
            data_range=1.0
        )

        scores.append(score)

    return float(np.mean(scores))


# ============================================================
# Create EDSR
# ============================================================

def create_edsr():

    model = EDSR(
        in_channels=4,
        out_channels=4,
        num_features=64,
        num_blocks=16,
        scale=4
    )

    checkpoint = torch.load(
        EDSR_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Create SwinIR
# ============================================================

def create_swinir():

    model = SwinIR(
        in_channels=4,
        out_channels=4,
        embed_dim=64,
        depths=4,
        num_heads=4,
        window_size=8,
        upscale=4
    )

    checkpoint = torch.load(
        SWINIR_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("SUPER-RESOLUTION MODEL EVALUATION")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Find test files
    # --------------------------------------------------------

    lr_files = sorted(
        glob.glob(
            os.path.join(
                TEST_LR_DIR,
                "*.tif"
            )
        )
    )

    hr_files = sorted(
        glob.glob(
            os.path.join(
                TEST_HR_DIR,
                "*.tif"
            )
        )
    )

    if len(lr_files) == 0:

        raise FileNotFoundError(
            "No LR test images found in: "
            + TEST_LR_DIR
        )

    if len(hr_files) == 0:

        raise FileNotFoundError(
            "No HR test images found in: "
            + TEST_HR_DIR
        )

    if len(lr_files) != len(hr_files):

        raise ValueError(
            f"LR images ({len(lr_files)}) and "
            f"HR images ({len(hr_files)}) do not match."
        )

    # Limit test samples
    lr_files = lr_files[:MAX_TEST_SAMPLES]
    hr_files = hr_files[:MAX_TEST_SAMPLES]

    print(
        "\nTest samples:",
        len(lr_files)
    )

    # --------------------------------------------------------
    # Load EDSR
    # --------------------------------------------------------

    print("\nLoading EDSR...")

    edsr = create_edsr()

    print("EDSR loaded successfully.")

    # --------------------------------------------------------
    # Load SwinIR
    # --------------------------------------------------------

    print("\nLoading SwinIR...")

    swinir = create_swinir()

    print("SwinIR loaded successfully.")

    # --------------------------------------------------------
    # Result storage
    # --------------------------------------------------------

    bicubic_psnr = []
    bicubic_ssim = []

    edsr_psnr = []
    edsr_ssim = []

    swinir_psnr = []
    swinir_ssim = []

    # ========================================================
    # Evaluation Loop
    # ========================================================

    for index, (lr_path, hr_path) in enumerate(
        zip(lr_files, hr_files)
    ):

        print(
            f"\nProcessing sample "
            f"{index + 1}/{len(lr_files)}"
        )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        lr = load_tiff(lr_path)
        hr = load_tiff(hr_path)

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        lr = normalize(lr)
        hr = normalize(hr)

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        lr_input = lr.unsqueeze(0).to(DEVICE)
        hr_input = hr.unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # Bicubic
        # ----------------------------------------------------

        bicubic = F.interpolate(
            lr_input,
            size=(
                hr.shape[1],
                hr.shape[2]
            ),
            mode="bicubic",
            align_corners=False
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

            edsr_output = edsr(
                lr_input
            )

        edsr_output = torch.clamp(
            edsr_output,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # SwinIR
        # ----------------------------------------------------

        with torch.no_grad():

            swinir_output = swinir(
                lr_input
            )

        swinir_output = torch.clamp(
            swinir_output,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Check shapes
        # ----------------------------------------------------

        if index == 0:

            print("\nImage shapes:")

            print(
                "LR      :",
                tuple(lr_input.shape)
            )

            print(
                "Bicubic :",
                tuple(bicubic.shape)
            )

            print(
                "EDSR    :",
                tuple(edsr_output.shape)
            )

            print(
                "SwinIR  :",
                tuple(swinir_output.shape)
            )

            print(
                "HR      :",
                tuple(hr_input.shape)
            )

        # ----------------------------------------------------
        # Calculate Metrics
        # ----------------------------------------------------

        b_psnr = calculate_psnr(
            bicubic[0],
            hr_input[0]
        )

        b_ssim = calculate_ssim(
            bicubic[0],
            hr_input[0]
        )

        e_psnr = calculate_psnr(
            edsr_output[0],
            hr_input[0]
        )

        e_ssim = calculate_ssim(
            edsr_output[0],
            hr_input[0]
        )

        s_psnr = calculate_psnr(
            swinir_output[0],
            hr_input[0]
        )

        s_ssim = calculate_ssim(
            swinir_output[0],
            hr_input[0]
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        bicubic_psnr.append(b_psnr)
        bicubic_ssim.append(b_ssim)

        edsr_psnr.append(e_psnr)
        edsr_ssim.append(e_ssim)

        swinir_psnr.append(s_psnr)
        swinir_ssim.append(s_ssim)

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Bicubic : "
            f"PSNR = {b_psnr:.4f} dB | "
            f"SSIM = {b_ssim:.4f}"
        )

        print(
            f"EDSR    : "
            f"PSNR = {e_psnr:.4f} dB | "
            f"SSIM = {e_ssim:.4f}"
        )

        print(
            f"SwinIR  : "
            f"PSNR = {s_psnr:.4f} dB | "
            f"SSIM = {s_ssim:.4f}"
        )

    # ========================================================
    # Average Results
    # ========================================================

    avg_b_psnr = float(
        np.mean(bicubic_psnr)
    )

    avg_b_ssim = float(
        np.mean(bicubic_ssim)
    )

    avg_e_psnr = float(
        np.mean(edsr_psnr)
    )

    avg_e_ssim = float(
        np.mean(edsr_ssim)
    )

    avg_s_psnr = float(
        np.mean(swinir_psnr)
    )

    avg_s_ssim = float(
        np.mean(swinir_ssim)
    )

    # ========================================================
    # Final Comparison
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        f"\nBicubic : "
        f"PSNR = {avg_b_psnr:.4f} dB | "
        f"SSIM = {avg_b_ssim:.4f}"
    )

    print(
        f"EDSR    : "
        f"PSNR = {avg_e_psnr:.4f} dB | "
        f"SSIM = {avg_e_ssim:.4f}"
    )

    print(
        f"SwinIR  : "
        f"PSNR = {avg_s_psnr:.4f} dB | "
        f"SSIM = {avg_s_ssim:.4f}"
    )

    # ========================================================
    # Improvement
    # ========================================================

    edsr_psnr_improvement = (
        avg_e_psnr - avg_b_psnr
    )

    edsr_ssim_improvement = (
        avg_e_ssim - avg_b_ssim
    )

    swinir_psnr_improvement = (
        avg_s_psnr - avg_b_psnr
    )

    swinir_ssim_improvement = (
        avg_s_ssim - avg_b_ssim
    )

    print("\nImprovements over Bicubic:")

    print(
        f"EDSR PSNR   : "
        f"{edsr_psnr_improvement:+.4f} dB"
    )

    print(
        f"EDSR SSIM   : "
        f"{edsr_ssim_improvement:+.4f}"
    )

    print(
        f"SwinIR PSNR : "
        f"{swinir_psnr_improvement:+.4f} dB"
    )

    print(
        f"SwinIR SSIM : "
        f"{swinir_ssim_improvement:+.4f}"
    )

    # ========================================================
    # Best Model
    # ========================================================

    if avg_s_psnr > avg_e_psnr:
        best_model = "SwinIR"
    else:
        best_model = "EDSR"

    print(
        "\nBest deep-learning model based on PSNR:",
        best_model
    )

    # ========================================================
    # Save Results
    # ========================================================

    os.makedirs(
        "results",
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "w"
    ) as file:

        file.write(
            "SUPER-RESOLUTION MODEL COMPARISON\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Number of test samples: "
            f"{len(lr_files)}\n\n"
        )

        file.write(
            "Method\tPSNR (dB)\tSSIM\n"
        )

        file.write(
            f"Bicubic\t"
            f"{avg_b_psnr:.4f}\t"
            f"{avg_b_ssim:.4f}\n"
        )

        file.write(
            f"EDSR\t"
            f"{avg_e_psnr:.4f}\t"
            f"{avg_e_ssim:.4f}\n"
        )

        file.write(
            f"SwinIR\t"
            f"{avg_s_psnr:.4f}\t"
            f"{avg_s_ssim:.4f}\n"
        )

        file.write(
            "\nImprovements over Bicubic\n"
        )

        file.write(
            f"EDSR PSNR: "
            f"{edsr_psnr_improvement:+.4f} dB\n"
        )

        file.write(
            f"EDSR SSIM: "
            f"{edsr_ssim_improvement:+.4f}\n"
        )

        file.write(
            f"SwinIR PSNR: "
            f"{swinir_psnr_improvement:+.4f} dB\n"
        )

        file.write(
            f"SwinIR SSIM: "
            f"{swinir_ssim_improvement:+.4f}\n"
        )

        file.write(
            f"\nBest deep-learning model: "
            f"{best_model}\n"
        )

    # ========================================================
    # Finished
    # ========================================================

    print(
        "\nResults saved to:"
    )

    print(
        RESULT_FILE
    )

    print("\nEvaluation completed.")

    print("=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()