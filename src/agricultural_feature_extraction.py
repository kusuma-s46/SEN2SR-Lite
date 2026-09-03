import os
import glob
import numpy as np


# ============================================================
# AGRICULTURAL FEATURE EXTRACTION
# ============================================================

NDVI_DIR = "results/ndvi"
AGRICULTURAL_DIR = "results/agricultural_area"

OUTPUT_DIR = "results/agricultural_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Extract features from one image
# ------------------------------------------------------------
def extract_features(ndvi_file):

    # Get image name
    base_name = os.path.basename(
        ndvi_file
    ).replace("_ndvi.npy", "")

    # Load NDVI
    ndvi = np.load(ndvi_file)

    # --------------------------------------------------------
    # Load agricultural classification
    # --------------------------------------------------------

    agricultural_file = os.path.join(
        AGRICULTURAL_DIR,
        base_name + "_agricultural.npy"
    )

    if os.path.exists(agricultural_file):

        agricultural = np.load(
            agricultural_file
        )

    else:

        print(
            f"Warning: Agricultural file not found for {base_name}"
        )

        # Fallback: detect agricultural vegetation
        agricultural = (
            ndvi >= 0.2
        ).astype(np.uint8)

    # --------------------------------------------------------
    # Total pixels
    # --------------------------------------------------------

    total_pixels = ndvi.size

    # --------------------------------------------------------
    # Agricultural pixels
    # --------------------------------------------------------

    agricultural_pixels = np.sum(
        agricultural == 1
    )

    agricultural_percentage = (
        agricultural_pixels /
        total_pixels
    ) * 100

    # --------------------------------------------------------
    # NDVI statistics
    # --------------------------------------------------------

    mean_ndvi = np.mean(ndvi)

    min_ndvi = np.min(ndvi)

    max_ndvi = np.max(ndvi)

    std_ndvi = np.std(ndvi)

    median_ndvi = np.median(ndvi)

    # --------------------------------------------------------
    # Vegetation classes
    # --------------------------------------------------------

    healthy_pixels = np.sum(
        ndvi >= 0.4
    )

    moderate_pixels = np.sum(
        (ndvi >= 0.2) &
        (ndvi < 0.4)
    )

    low_vegetation_pixels = np.sum(
        (ndvi >= 0.1) &
        (ndvi < 0.2)
    )

    dry_pixels = np.sum(
        (ndvi >= 0.0) &
        (ndvi < 0.1)
    )

    water_pixels = np.sum(
        ndvi < 0.0
    )

    # --------------------------------------------------------
    # Percentages
    # --------------------------------------------------------

    healthy_percentage = (
        healthy_pixels /
        total_pixels
    ) * 100

    moderate_percentage = (
        moderate_pixels /
        total_pixels
    ) * 100

    low_vegetation_percentage = (
        low_vegetation_pixels /
        total_pixels
    ) * 100

    dry_percentage = (
        dry_pixels /
        total_pixels
    ) * 100

    water_percentage = (
        water_pixels /
        total_pixels
    ) * 100

    # --------------------------------------------------------
    # Vegetation level
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
    # Agricultural condition
    # --------------------------------------------------------

    if agricultural_percentage >= 70:

        agricultural_condition = "HIGH"

    elif agricultural_percentage >= 40:

        agricultural_condition = "MODERATE"

    elif agricultural_percentage >= 20:

        agricultural_condition = "LOW"

    else:

        agricultural_condition = "VERY LOW"

    # --------------------------------------------------------
    # Crop growth indication
    # --------------------------------------------------------

    if healthy_percentage >= 60:

        crop_growth = "HEALTHY / HIGH VEGETATION"

    elif healthy_percentage >= 30:

        crop_growth = "MODERATE VEGETATION"

    elif moderate_percentage >= 30:

        crop_growth = "EARLY / DEVELOPING VEGETATION"

    else:

        crop_growth = "LOW VEGETATION / POSSIBLE DRY CONDITION"

    # --------------------------------------------------------
    # Create feature dictionary
    # --------------------------------------------------------

    features = {

        "image": base_name,

        "mean_ndvi": mean_ndvi,

        "median_ndvi": median_ndvi,

        "min_ndvi": min_ndvi,

        "max_ndvi": max_ndvi,

        "std_ndvi": std_ndvi,

        "agricultural_percentage":
            agricultural_percentage,

        "healthy_vegetation_percentage":
            healthy_percentage,

        "moderate_vegetation_percentage":
            moderate_percentage,

        "low_vegetation_percentage":
            low_vegetation_percentage,

        "dry_barren_percentage":
            dry_percentage,

        "water_percentage":
            water_percentage,

        "vegetation_level":
            vegetation_level,

        "agricultural_condition":
            agricultural_condition,

        "crop_growth_indication":
            crop_growth
    }

    return features


# ------------------------------------------------------------
# Main
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
            "\nRun:"
        )

        print(
            "python src/ndvi_analysis.py"
        )

        return

    print(
        "\nFound NDVI files:",
        len(ndvi_files)
    )

    all_features = []

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for ndvi_file in ndvi_files:

        features = extract_features(
            ndvi_file
        )

        all_features.append(
            features
        )

        print("\n" + "=" * 65)
        print("AGRICULTURAL FEATURES")
        print("=" * 65)

        print(
            "Image:",
            features["image"]
        )

        print(
            f"Mean NDVI: "
            f"{features['mean_ndvi']:.4f}"
        )

        print(
            f"NDVI Range: "
            f"{features['min_ndvi']:.4f} → "
            f"{features['max_ndvi']:.4f}"
        )

        print(
            f"Agricultural Area: "
            f"{features['agricultural_percentage']:.2f}%"
        )

        print(
            f"Healthy Vegetation: "
            f"{features['healthy_vegetation_percentage']:.2f}%"
        )

        print(
            f"Moderate Vegetation: "
            f"{features['moderate_vegetation_percentage']:.2f}%"
        )

        print(
            f"Low Vegetation: "
            f"{features['low_vegetation_percentage']:.2f}%"
        )

        print(
            f"Dry / Barren: "
            f"{features['dry_barren_percentage']:.2f}%"
        )

        print(
            f"Water: "
            f"{features['water_percentage']:.2f}%"
        )

        print(
            "Vegetation Level:",
            features["vegetation_level"]
        )

        print(
            "Agricultural Condition:",
            features["agricultural_condition"]
        )

        print(
            "Crop Growth Indication:",
            features["crop_growth_indication"]
        )

        print("=" * 65)

    # --------------------------------------------------------
    # Save CSV-style feature file
    # --------------------------------------------------------

    feature_file = os.path.join(
        OUTPUT_DIR,
        "agricultural_features.csv"
    )

    headers = list(
        all_features[0].keys()
    )

    with open(
        feature_file,
        "w"
    ) as f:

        f.write(
            ",".join(headers) + "\n"
        )

        for features in all_features:

            row = []

            for header in headers:

                value = features[header]

                if isinstance(
                    value,
                    float
                ):

                    row.append(
                        f"{value:.6f}"
                    )

                else:

                    row.append(
                        str(value)
                    )

            f.write(
                ",".join(row) + "\n"
            )

    # --------------------------------------------------------
    # Save readable report
    # --------------------------------------------------------

    report_file = os.path.join(
        OUTPUT_DIR,
        "agricultural_features_report.txt"
    )

    with open(
        report_file,
        "w"
    ) as f:

        f.write(
            "AGRICULTURAL FEATURE EXTRACTION REPORT\n"
        )

        f.write(
            "=" * 65 + "\n\n"
        )

        for features in all_features:

            f.write(
                f"Image: {features['image']}\n"
            )

            f.write(
                f"Mean NDVI: "
                f"{features['mean_ndvi']:.4f}\n"
            )

            f.write(
                f"Median NDVI: "
                f"{features['median_ndvi']:.4f}\n"
            )

            f.write(
                f"Minimum NDVI: "
                f"{features['min_ndvi']:.4f}\n"
            )

            f.write(
                f"Maximum NDVI: "
                f"{features['max_ndvi']:.4f}\n"
            )

            f.write(
                f"NDVI Standard Deviation: "
                f"{features['std_ndvi']:.4f}\n"
            )

            f.write(
                f"Agricultural Area: "
                f"{features['agricultural_percentage']:.2f}%\n"
            )

            f.write(
                f"Healthy Vegetation: "
                f"{features['healthy_vegetation_percentage']:.2f}%\n"
            )

            f.write(
                f"Moderate Vegetation: "
                f"{features['moderate_vegetation_percentage']:.2f}%\n"
            )

            f.write(
                f"Low Vegetation: "
                f"{features['low_vegetation_percentage']:.2f}%\n"
            )

            f.write(
                f"Dry / Barren: "
                f"{features['dry_barren_percentage']:.2f}%\n"
            )

            f.write(
                f"Water: "
                f"{features['water_percentage']:.2f}%\n"
            )

            f.write(
                f"Vegetation Level: "
                f"{features['vegetation_level']}\n"
            )

            f.write(
                f"Agricultural Condition: "
                f"{features['agricultural_condition']}\n"
            )

            f.write(
                f"Crop Growth Indication: "
                f"{features['crop_growth_indication']}\n"
            )

            f.write(
                "\n" + "-" * 65 + "\n\n"
            )

    print("\n" + "=" * 65)
    print(
        "AGRICULTURAL FEATURE EXTRACTION COMPLETED"
    )
    print("=" * 65)

    print(
        "\nFeatures saved to:"
    )

    print(
        "results/agricultural_features/"
    )

    print(
        "\nFeature file:"
    )

    print(
        "results/agricultural_features/"
        "agricultural_features.csv"
    )

    print(
        "\nReport:"
    )

    print(
        "results/agricultural_features/"
        "agricultural_features_report.txt"
    )


if __name__ == "__main__":
    main()