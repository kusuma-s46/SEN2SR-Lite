import os
import glob
import torch
import numpy as np
import rasterio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from edsr_model import EDSR
from swinir_model import SwinIR
from esrgan_model import ESRGAN


# ============================================================
# SETTINGS
# ============================================================

TEST_LR_DIR = "dataset/local/test/lr"
TEST_HR_DIR = "dataset/local/test/hr"

EDSR_PATH = "models/edsr_local.pth"
SWINIR_PATH = "models/swinir_local.pth"
ESRGAN_PATH = "models/esrgan_local.pth"

OUTPUT_FILE = "results/final_sr_model_comparison.txt"

MAX_TEST_SAMPLES = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("FINAL SUPER-RESOLUTION MODEL COMPARISON")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# LOAD TIFF
# ============================================================

def load_tiff(path):

    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)

    return image


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(image):

    return image / 5000.0


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(pred, target):

    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)

    return peak_signal_noise_ratio(
        target,
        pred,
        data_range=1.0
    )


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(pred, target):

    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)

    scores = []

    for band in range(pred.shape[0]):

        score = structural_similarity(
            target[band],
            pred[band],
            data_range=1.0
        )

        scores.append(score)

    return np.mean(scores)


# ============================================================
# GET TEST FILES
# ============================================================

lr_files = sorted(
    glob.glob(os.path.join(TEST_LR_DIR, "*.tif"))
)

hr_files = sorted(
    glob.glob(os.path.join(TEST_HR_DIR, "*.tif"))
)

lr_files = lr_files[:MAX_TEST_SAMPLES]
hr_files = hr_files[:MAX_TEST_SAMPLES]

print()
print("Test samples:", len(lr_files))


# ============================================================
# LOAD EDSR
# ============================================================

print()
print("Loading EDSR...")

edsr = EDSR(
    in_channels=4,
    out_channels=4,
    num_features=64,
    num_blocks=16,
    scale=4
)

edsr.load_state_dict(
    torch.load(
        EDSR_PATH,
        map_location=DEVICE
    )
)

edsr = edsr.to(DEVICE)
edsr.eval()


# ============================================================
# LOAD SWINIR
# ============================================================

print("Loading SwinIR...")

swinir = SwinIR(
    in_channels=4,
    out_channels=4,
    embed_dim=64,
    depths=4,
    num_heads=4,
    window_size=8,
    upscale=4
)

swinir.load_state_dict(
    torch.load(
        SWINIR_PATH,
        map_location=DEVICE
    )
)

swinir = swinir.to(DEVICE)
swinir.eval()


# ============================================================
# LOAD ESRGAN
# ============================================================

print("Loading ESRGAN...")

esrgan = ESRGAN(
    in_channels=4,
    out_channels=4,
    num_features=64,
    num_blocks=6,
    growth_channels=32,
    scale=4
)

esrgan.load_state_dict(
    torch.load(
        ESRGAN_PATH,
        map_location=DEVICE
    )
)

esrgan = esrgan.to(DEVICE)
esrgan.eval()


# ============================================================
# RESULTS STORAGE
# ============================================================

results = {

    "Bicubic": {
        "psnr": [],
        "ssim": []
    },

    "EDSR": {
        "psnr": [],
        "ssim": []
    },

    "SwinIR": {
        "psnr": [],
        "ssim": []
    },

    "ESRGAN": {
        "psnr": [],
        "ssim": []
    }
}


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 70)
print("EVALUATING MODELS")
print("=" * 70)


with torch.no_grad():

    for i, (lr_path, hr_path) in enumerate(
        zip(lr_files, hr_files)
    ):

        print()
        print(
            f"Sample {i + 1}/{len(lr_files)}"
        )

        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        lr = load_tiff(lr_path)
        hr = load_tiff(hr_path)

        lr = normalize(lr)
        hr = normalize(hr)

        lr_tensor = torch.from_numpy(
            lr
        ).unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # Bicubic
        # ----------------------------------------------------

        bicubic = torch.nn.functional.interpolate(
            lr_tensor,
            size=(520, 520),
            mode="bicubic",
            align_corners=False
        )

        # ----------------------------------------------------
        # EDSR
        # ----------------------------------------------------

        edsr_output = edsr(
            lr_tensor
        )

        # ----------------------------------------------------
        # SwinIR
        # ----------------------------------------------------

        swinir_output = swinir(
            lr_tensor
        )

        # ----------------------------------------------------
        # ESRGAN
        # ----------------------------------------------------

        esrgan_output = esrgan(
            lr_tensor
        )

        # ----------------------------------------------------
        # Convert tensors to numpy
        # ----------------------------------------------------

        bicubic = bicubic.squeeze(0).cpu().numpy()

        edsr_output = (
            edsr_output
            .squeeze(0)
            .cpu()
            .numpy()
        )

        swinir_output = (
            swinir_output
            .squeeze(0)
            .cpu()
            .numpy()
        )

        esrgan_output = (
            esrgan_output
            .squeeze(0)
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        bicubic_psnr = calculate_psnr(
            bicubic,
            hr
        )

        bicubic_ssim = calculate_ssim(
            bicubic,
            hr
        )

        edsr_psnr = calculate_psnr(
            edsr_output,
            hr
        )

        edsr_ssim = calculate_ssim(
            edsr_output,
            hr
        )

        swinir_psnr = calculate_psnr(
            swinir_output,
            hr
        )

        swinir_ssim = calculate_ssim(
            swinir_output,
            hr
        )

        esrgan_psnr = calculate_psnr(
            esrgan_output,
            hr
        )

        esrgan_ssim = calculate_ssim(
            esrgan_output,
            hr
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results["Bicubic"]["psnr"].append(
            bicubic_psnr
        )

        results["Bicubic"]["ssim"].append(
            bicubic_ssim
        )

        results["EDSR"]["psnr"].append(
            edsr_psnr
        )

        results["EDSR"]["ssim"].append(
            edsr_ssim
        )

        results["SwinIR"]["psnr"].append(
            swinir_psnr
        )

        results["SwinIR"]["ssim"].append(
            swinir_ssim
        )

        results["ESRGAN"]["psnr"].append(
            esrgan_psnr
        )

        results["ESRGAN"]["ssim"].append(
            esrgan_ssim
        )

        print(
            f"Bicubic : PSNR={bicubic_psnr:.4f} "
            f"SSIM={bicubic_ssim:.4f}"
        )

        print(
            f"EDSR    : PSNR={edsr_psnr:.4f} "
            f"SSIM={edsr_ssim:.4f}"
        )

        print(
            f"SwinIR  : PSNR={swinir_psnr:.4f} "
            f"SSIM={swinir_ssim:.4f}"
        )

        print(
            f"ESRGAN  : PSNR={esrgan_psnr:.4f} "
            f"SSIM={esrgan_ssim:.4f}"
        )


# ============================================================
# FINAL AVERAGES
# ============================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

final_results = {}

for model_name in results:

    avg_psnr = np.mean(
        results[model_name]["psnr"]
    )

    avg_ssim = np.mean(
        results[model_name]["ssim"]
    )

    final_results[model_name] = {
        "psnr": avg_psnr,
        "ssim": avg_ssim
    }

    print()
    print(model_name)

    print(
        f"Average PSNR : {avg_psnr:.4f} dB"
    )

    print(
        f"Average SSIM : {avg_ssim:.4f}"
    )


# ============================================================
# FIND BEST MODEL
# ============================================================

best_model = max(
    ["EDSR", "SwinIR", "ESRGAN"],
    key=lambda x: final_results[x]["psnr"]
)

print()
print("=" * 70)
print("BEST SUPER-RESOLUTION MODEL")
print("=" * 70)

print("Best model:", best_model)

print(
    "PSNR:",
    f"{final_results[best_model]['psnr']:.4f} dB"
)

print(
    "SSIM:",
    f"{final_results[best_model]['ssim']:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w"
) as f:

    f.write(
        "FINAL SUPER-RESOLUTION MODEL COMPARISON\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Test Samples: {len(lr_files)}\n"
    )

    f.write(
        f"Device: {DEVICE}\n\n"
    )

    for model_name in final_results:

        f.write(
            f"{model_name}\n"
        )

        f.write(
            f"Average PSNR: "
            f"{final_results[model_name]['psnr']:.4f} dB\n"
        )

        f.write(
            f"Average SSIM: "
            f"{final_results[model_name]['ssim']:.4f}\n\n"
        )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        f"BEST MODEL: {best_model}\n"
    )

    f.write(
        f"BEST PSNR: "
        f"{final_results[best_model]['psnr']:.4f} dB\n"
    )

    f.write(
        f"BEST SSIM: "
        f"{final_results[best_model]['ssim']:.4f}\n"
    )


print()
print(
    "Results saved to:",
    OUTPUT_FILE
)

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)