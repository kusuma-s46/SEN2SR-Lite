import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DRY / BARREN LAND DETECTION
# ============================================================

NDVI_DIR = "results/ndvi"
OUTPUT_DIR = "results/dry_barren"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# NDVI Classification
# ------------------------------------------------------------
def classify_ndvi(ndvi):
    """
    Classify pixels based on NDVI value.

    Classes:
    0 = Water / Very Low Vegetation
    1 = Dry / Barren Land
    2 = Very Low Vegetation
    3 = Sparse Vegetation
    4 = Vegetated Area
    """

    classification = np.zeros_like(ndvi, dtype=np.uint8)

    # Water / very low vegetation
    classification[ndvi < 0.0] = 0

    # Dry / barren land
    classification[(ndvi >= 0.0) & (ndvi < 0.1)] = 1

    # Very low vegetation
    classification[(ndvi >= 0.1) & (ndvi < 0.2)] = 2

    # Sparse vegetation
    classification[(ndvi >= 0.2) & (ndvi < 0.4)] = 3

    # Vegetated area
    classification[ndvi >= 0.4] = 4

    return classification


# ------------------------------------------------------------
# Process One Image
# ------------------------------------------------------------
def process_image(ndvi_file):

    ndvi = np.load(ndvi_file)

    classification = classify_ndvi(ndvi)

    total_pixels = classification.size

    water_pixels = np.sum(classification == 0)
    dry_pixels = np.sum(classification == 1)
    very_low_pixels = np.sum(classification == 2)
    sparse_pixels = np.sum(classification == 3)
    vegetation_pixels = np.sum(classification == 4)

    # Percentages
    water_percentage = (water_pixels / total_pixels) * 100
    dry_percentage = (dry_pixels / total_pixels) * 100
    very_low_percentage = (very_low_pixels / total_pixels) * 100
    sparse_percentage = (sparse_pixels / total_pixels) * 100
    vegetation_percentage = (vegetation_pixels / total_pixels) * 100

    mean_ndvi = np.mean(ndvi)

    # --------------------------------------------------------
    # Save classification array
    # --------------------------------------------------------

    base_name = os.path.basename(ndvi_file).replace("_ndvi.npy", "")

    np.save(
        os.path.join(
            OUTPUT_DIR,
            base_name + "_dry_barren.npy"
        ),
        classification
    )

    # --------------------------------------------------------
    # Create visualization
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.imshow(classification)

    plt.title("Dry / Barren Land Analysis")

    plt.colorbar(
        ticks=[0, 1, 2, 3, 4],
        label="Land / Vegetation Class"
    )

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    output_image = os.path.join(
        OUTPUT_DIR,
        base_name + "_dry_barren.png"
    )

    plt.savefig(
        output_image,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Print Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DRY / BARREN LAND ANALYSIS")
    print("=" * 60)

    print("Image:", base_name)

    print(f"Mean NDVI              : {mean_ndvi:.4f}")

    print(f"Water / Very Low       : {water_percentage:.2f}%")
    print(f"Dry / Barren Land      : {dry_percentage:.2f}%")
    print(f"Very Low Vegetation    : {very_low_percentage:.2f}%")
    print(f"Sparse Vegetation      : {sparse_percentage:.2f}%")
    print(f"Vegetated Area         : {vegetation_percentage:.2f}%")

    # --------------------------------------------------------
    # Determine dominant condition
    # --------------------------------------------------------

    if dry_percentage >= 30:
        condition = "HIGH DRY / BARREN AREA"

    elif dry_percentage >= 15:
        condition = "MODERATE DRY / BARREN AREA"

    else:
        condition = "LOW DRY / BARREN AREA"

    print("Overall Condition      :", condition)

    print("=" * 60)

    return {
        "image": base_name,
        "mean_ndvi": mean_ndvi,
        "water_percentage": water_percentage,
        "dry_percentage": dry_percentage,
        "very_low_percentage": very_low_percentage,
        "sparse_percentage": sparse_percentage,
        "vegetation_percentage": vegetation_percentage,
        "condition": condition
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

        print("ERROR: No NDVI files found.")

        print(
            "\nPlease run NDVI analysis first:"
        )

        print(
            "python src/ndvi_analysis.py"
        )

        return

    print("\nFound NDVI files:", len(ndvi_files))

    results = []

    # Process all available NDVI images
    for ndvi_file in ndvi_files:

        result = process_image(ndvi_file)

        results.append(result)

    # --------------------------------------------------------
    # Save Report
    # --------------------------------------------------------

    report_file = os.path.join(
        OUTPUT_DIR,
        "dry_barren_report.txt"
    )

    with open(report_file, "w") as f:

        f.write("DRY / BARREN LAND ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        for result in results:

            f.write(
                f"Image: {result['image']}\n"
            )

            f.write(
                f"Mean NDVI: {result['mean_ndvi']:.4f}\n"
            )

            f.write(
                f"Water / Very Low: "
                f"{result['water_percentage']:.2f}%\n"
            )

            f.write(
                f"Dry / Barren Land: "
                f"{result['dry_percentage']:.2f}%\n"
            )

            f.write(
                f"Very Low Vegetation: "
                f"{result['very_low_percentage']:.2f}%\n"
            )

            f.write(
                f"Sparse Vegetation: "
                f"{result['sparse_percentage']:.2f}%\n"
            )

            f.write(
                f"Vegetated Area: "
                f"{result['vegetation_percentage']:.2f}%\n"
            )

            f.write(
                f"Overall Condition: "
                f"{result['condition']}\n"
            )

            f.write("\n" + "-" * 60 + "\n\n")

    print("\n" + "=" * 60)
    print("DRY / BARREN ANALYSIS COMPLETED")
    print("=" * 60)

    print("\nResults saved in:")
    print("results/dry_barren/")

    print("\nReport:")
    print("results/dry_barren/dry_barren_report.txt")


if __name__ == "__main__":
    main()