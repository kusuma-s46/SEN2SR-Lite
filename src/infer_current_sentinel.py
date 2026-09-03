import os
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt

from rasterio.transform import Affine

from edsr_model import EDSR


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "dataset/current/input/current_sentinel.tif"

MODEL_PATH = "models/edsr_local.pth"

OUTPUT_DIR = "results/current_inference"

OUTPUT_TIF = os.path.join(
    OUTPUT_DIR,
    "current_sentinel_SR.tif"
)

OUTPUT_RGB = os.path.join(
    OUTPUT_DIR,
    "current_sentinel_RGB.png"
)

OUTPUT_NDVI = os.path.join(
    OUTPUT_DIR,
    "current_sentinel_NDVI.png"
)

# Sentinel-2 B02, B03, B04, B08
BAND_NAMES = ["B02", "B03", "B04", "B08"]

# Same normalization used during our training
NORMALIZATION_VALUE = 5000.0

# Super-resolution scale
SCALE = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD EDSR MODEL
# ============================================================

def load_model():

    print("Loading trained EDSR model...")

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

    # Handle checkpoints saved either as state_dict
    # or as a dictionary containing model_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")
    print("Device:", DEVICE)

    return model


# ============================================================
# LOAD CURRENT SENTINEL-2 IMAGE
# ============================================================

def load_input_image():

    print("\nReading current Sentinel-2 image...")
    print("Input:", INPUT_PATH)

    with rasterio.open(INPUT_PATH) as src:

        image = src.read().astype(np.float32)

        profile = src.profile.copy()

        transform = src.transform
        crs = src.crs

        width = src.width
        height = src.height

        print("Input shape:", image.shape)
        print("Input size:", width, "x", height)
        print("Number of bands:", src.count)
        print("CRS:", crs)

    if image.shape[0] != 4:

        raise ValueError(
            "Input image must contain exactly 4 bands: "
            "B02, B03, B04, B08"
        )

    return image, profile, transform, crs


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess(image):

    print("\nPreprocessing image...")

    # Convert reflectance values to 0-1 range
    image = image / NORMALIZATION_VALUE

    # Keep values within valid range
    image = np.clip(image, 0.0, 1.0)

    # Convert:
    # [4, H, W]
    # to
    # [1, 4, H, W]

    tensor = torch.from_numpy(image).float()

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    print("Model input shape:", tensor.shape)

    return tensor


# ============================================================
# SUPER-RESOLUTION
# ============================================================

def generate_sr(model, input_tensor):

    print("\nGenerating super-resolved image...")

    with torch.no_grad():

        sr = model(input_tensor)

    sr = sr.squeeze(0).cpu().numpy()

    # Keep output between 0 and 1
    sr = np.clip(sr, 0.0, 1.0)

    print("SR output shape:", sr.shape)

    return sr


# ============================================================
# SAVE SR GEOTIFF
# ============================================================

def save_sr_geotiff(
    sr,
    profile,
    transform,
    crs
):

    print("\nSaving super-resolved GeoTIFF...")

    bands, sr_height, sr_width = sr.shape

    # Convert back to reflectance-like values
    sr_output = sr * NORMALIZATION_VALUE

    # Update geospatial transform for 4x resolution
    new_transform = transform * Affine.scale(
        1.0 / SCALE,
        1.0 / SCALE
    )

    new_profile = profile.copy()

    new_profile.update(
        {
            "driver": "GTiff",
            "height": sr_height,
            "width": sr_width,
            "count": bands,
            "dtype": "float32",
            "transform": new_transform,
            "crs": crs
        }
    )

    with rasterio.open(
        OUTPUT_TIF,
        "w",
        **new_profile
    ) as dst:

        dst.write(sr_output.astype(np.float32))

        # Store band names
        for i, band_name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(i, band_name)

    print("Saved:")
    print(OUTPUT_TIF)


# ============================================================
# CREATE RGB VISUALIZATION
# ============================================================

def create_rgb_visualization(sr):

    print("\nCreating RGB visualization...")

    # Band order:
    # 0 = B02 Blue
    # 1 = B03 Green
    # 2 = B04 Red
    # 3 = B08 NIR

    blue = sr[0]
    green = sr[1]
    red = sr[2]

    rgb = np.stack(
        [red, green, blue],
        axis=-1
    )

    # Percentile stretching
    lower = np.percentile(rgb, 2)
    upper = np.percentile(rgb, 98)

    rgb = (rgb - lower) / (upper - lower + 1e-8)

    rgb = np.clip(rgb, 0, 1)

    plt.figure(figsize=(10, 10))

    plt.imshow(rgb)

    plt.title(
        "SEN2SR-Lite - Current Sentinel-2 Super-Resolved Image"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_RGB,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:")
    print(OUTPUT_RGB)


# ============================================================
# NDVI
# ============================================================

def create_ndvi(sr):

    print("\nCalculating NDVI...")

    # B04 = Red
    # B08 = NIR

    red = sr[2]
    nir = sr[3]

    denominator = nir + red

    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(nir),
        where=denominator != 0
    )

    ndvi = np.clip(
        ndvi,
        -1,
        1
    )

    plt.figure(figsize=(10, 10))

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
        "NDVI from Current Super-Resolved Sentinel-2 Image"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_NDVI,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:")
    print(OUTPUT_NDVI)

    print(
        "Mean NDVI:",
        round(float(np.mean(ndvi)), 4)
    )

    return ndvi


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SEN2SR-Lite - CURRENT SENTINEL-2 IMAGE INFERENCE")
    print("=" * 60)

    # Check input
    if not os.path.exists(INPUT_PATH):

        print("\nERROR: Current Sentinel-2 image not found.")

        print("\nPlease put your image here:")

        print(
            "dataset/current/input/current_sentinel.tif"
        )

        return

    # Check model
    if not os.path.exists(MODEL_PATH):

        print("\nERROR: Trained EDSR model not found.")

        print(
            "Expected:"
        )

        print(MODEL_PATH)

        return

    # Load model
    model = load_model()

    # Load image
    image, profile, transform, crs = load_input_image()

    # Preprocess
    input_tensor = preprocess(image)

    # Generate SR
    sr = generate_sr(
        model,
        input_tensor
    )

    # Save GeoTIFF
    save_sr_geotiff(
        sr,
        profile,
        transform,
        crs
    )

    # RGB visualization
    create_rgb_visualization(sr)

    # NDVI
    create_ndvi(sr)

    print("\n" + "=" * 60)
    print("CURRENT IMAGE INFERENCE COMPLETED")
    print("=" * 60)

    print("\nOutputs:")

    print("1. Super-resolved GeoTIFF:")
    print(OUTPUT_TIF)

    print("\n2. RGB visualization:")
    print(OUTPUT_RGB)

    print("\n3. NDVI map:")
    print(OUTPUT_NDVI)

    print("\n10 m Sentinel-2 → 2.5 m AI Super-Resolution")


if __name__ == "__main__":
    main()