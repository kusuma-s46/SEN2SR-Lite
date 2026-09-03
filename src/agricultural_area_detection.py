import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# AGRICULTURAL AREA DETECTION
# ============================================================

NDVI_DIR = "results/ndvi"
DRY_BARREN_DIR = "results/dry_barren"

OUTPUT_DIR = "results/agricultural_area"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Agricultural Classification
# ------------------------------------------------------------
def detect_agriculture(ndvi, dry_barren):

    agricultural = np.zeros_like(ndvi, dtype=np.uint8)

    # Vegetated regions
    vegetation = ndvi >= 0.2

    # Sparse vegetation that may still belong to agriculture
    sparse_vegetation = (
        (ndvi >= 0.2) &
        (ndvi < 0.4)
    )

    # Healthy vegetation
    healthy_vegetation = ndvi >= 0.4

    # Combine agricultural vegetation
    agricultural[
        sparse_vegetation | healthy_vegetation
    ] = 1

    return agricultural


# ------------------------------------------------------------
# Process One Image
# ------------------------------------------------------------
def process_image(ndvi_file):

    base_name = os.path.basename(
        ndvi_file
    ).replace("_ndvi.npy", "")

    ndvi = np.load(ndvi_file)

    # --------------------------------------------------------
    # Find corresponding dry/barren classification
    # --------------------------------------------------------

    dry_file = os.path.join(
        DRY_BARREN_DIR,
        base_name + "_dry_barren.npy"
    )

    if os.path.exists(dry_file):

        dry_barren = np.load(dry_file)

    else:

        print(
            f"Warning: Dry/barren file not found for {base_name}"
        )

        # Create empty classification if unavailable
        dry_barren = np.zeros_like(
            ndvi,
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # Agricultural Detection
    # --------------------------------------------------------

    agricultural = detect_agriculture(
        ndvi,
        dry_barren
    )

    total_pixels = agricultural.size

    agricultural_pixels = np.sum(
        agricultural == 1
    )

    non_agricultural_pixels = np.sum(
        agricultural == 0
    )

    agricultural_percentage = (
        agricultural_pixels /
        total_pixels
    ) * 100

    non_agricultural_percentage = (
        non_agricultural_pixels /
        total_pixels
    ) * 100

    # --------------------------------------------------------
    # Vegetation Statistics
    # --------------------------------------------------------

    vegetation_pixels = np.sum(
        ndvi >= 0.4
    )

    sparse_pixels = np.sum(
        (ndvi >= 0.2) &
        (ndvi < 0.4)
    )

    dry_pixels = np.sum(
        (ndvi >= 0.0) &
        (ndvi < 0.2)
    )

    vegetation_percentage = (
        vegetation_pixels /
        total_pixels
    ) * 100

    sparse_percentage = (
        sparse_pixels /
        total_pixels
    ) * 100

    dry_percentage = (
        dry_pixels /
        total_pixels
    ) * 100

    mean_ndvi = np.mean(ndvi)

    # --------------------------------------------------------
    # Vegetation Level
    # --------------------------------------------------------

    if mean_ndvi < 0.2:

        vegetation_level = "LOW"

    elif mean_ndvi < 0.4:

        vegetation_level = "MODERATE"

    elif mean_ndvi < 0.6:

        vegetation_level = "GOOD"

    else:

        vegetation_level = "HIGH"

    # --------------------------------------------------------
    # Agricultural Likelihood
    # --------------------------------------------------------

    if agricultural_percentage >= 70:

        agricultural_status = "HIGH AGRICULTURAL AREA"

    elif agricultural_percentage >= 40:

        agricultural_status = "MODERATE AGRICULTURAL AREA"

    elif agricultural_percentage >= 20:

        agricultural_status = "LOW AGRICULTURAL AREA"

    else:

        agricultural_status = "NON-AGRICULTURAL / LIMITED AGRICULTURE"

    # --------------------------------------------------------
    # Save Agricultural Classification
    # --------------------------------------------------------

    np.save(
        os.path.join(
            OUTPUT_DIR,
            base_name + "_agricultural.npy"
        ),
        agricultural
    )

    # --------------------------------------------------------
    # Create Agricultural Map
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.imshow(
        agricultural
    )

    plt.title(
        "Agricultural Area Detection"
    )

    plt.colorbar(
        ticks=[0, 1],
        label="Agricultural Class"
    )

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    map_file = os.path.join(
        OUTPUT_DIR,
        base_name + "_agricultural.png"
    )

    plt.savefig(
        map_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("AGRICULTURAL AREA DETECTION")
    print("=" * 65)

    print(
        "Image                  :",
        base_name
    )

    print(
        f"Mean NDVI              : {mean_ndvi:.4f}"
    )

    print(
        f"Agricultural Area      : "
        f"{agricultural_percentage:.2f}%"
    )

    print(
        f"Non-Agricultural Area  : "
        f"{non_agricultural_percentage:.2f}%"
    )

    print(
        f"Healthy Vegetation     : "
        f"{vegetation_percentage:.2f}%"
    )

    print(
        f"Sparse Vegetation      : "
        f"{sparse_percentage:.2f}%"
    )

    print(
        f"Dry / Low Vegetation   : "
        f"{dry_percentage:.2f}%"
    )

    print(
        "Vegetation Level       :",
        vegetation_level
    )

    print(
        "Agricultural Status    :",
        agricultural_status
    )

    print("=" * 65)

    return {
        "image": base_name,
        "mean_ndvi": mean_ndvi,
        "agricultural_percentage":
            agricultural_percentage,
        "non_agricultural_percentage":
            non_agricultural_percentage,
        "vegetation_percentage":
            vegetation_percentage,
        "sparse_percentage":
            sparse_percentage,
        "dry_percentage":
            dry_percentage,
        "vegetation_level":
            vegetation_level,
        "agricultural_status":
            agricultural_status
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    ndvi_files = sorted(
        glob.glob(
            os.path.join(
                NDVI_DIR,
                "*_ndvi.npy"
            )
        )
    )

    if len(ndvi_files) == 0:

        print(
            "ERROR: No NDVI files found."
        )

        print(
            "\nRun NDVI analysis first:"
        )

        print(
            "python src/ndvi_analysis.py"
        )

        return

    print(
        "\nNDVI files found:",
        len(ndvi_files)
    )

    results = []

    # --------------------------------------------------------
    # Process all images
    # --------------------------------------------------------

    for ndvi_file in ndvi_files:

        result = process_image(
            ndvi_file
        )

        results.append(result)

    # --------------------------------------------------------
    # Save Report
    # --------------------------------------------------------

    report_file = os.path.join(
        OUTPUT_DIR,
        "agricultural_area_report.txt"
    )

    with open(
        report_file,
        "w"
    ) as f:

        f.write(
            "AGRICULTURAL AREA DETECTION REPORT\n"
        )

        f.write(
            "=" * 65 + "\n\n"
        )

        for result in results:

            f.write(
                f"Image: {result['image']}\n"
            )

            f.write(
                f"Mean NDVI: "
                f"{result['mean_ndvi']:.4f}\n"
            )

            f.write(
                f"Agricultural Area: "
                f"{result['agricultural_percentage']:.2f}%\n"
            )

            f.write(
                f"Non-Agricultural Area: "
                f"{result['non_agricultural_percentage']:.2f}%\n"
            )

            f.write(
                f"Healthy Vegetation: "
                f"{result['vegetation_percentage']:.2f}%\n"
            )

            f.write(
                f"Sparse Vegetation: "
                f"{result['sparse_percentage']:.2f}%\n"
            )

            f.write(
                f"Dry / Low Vegetation: "
                f"{result['dry_percentage']:.2f}%\n"
            )

            f.write(
                f"Vegetation Level: "
                f"{result['vegetation_level']}\n"
            )

            f.write(
                f"Agricultural Status: "
                f"{result['agricultural_status']}\n"
            )

            f.write(
                "\n" + "-" * 65 + "\n\n"
            )

    print("\n" + "=" * 65)
    print(
        "AGRICULTURAL AREA DETECTION COMPLETED"
    )
    print("=" * 65)

    print(
        "\nResults saved in:"
    )

    print(
        "results/agricultural_area/"
    )

    print(
        "\nReport:"
    )

    print(
        "results/agricultural_area/"
        "agricultural_area_report.txt"
    )


if __name__ == "__main__":
    main()