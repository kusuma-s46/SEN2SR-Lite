import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

NDVI_DIR = "results/ndvi"

OUTPUT_DIR = "results/vegetation_health"

# Process maximum number of NDVI images
MAX_SAMPLES = 20


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# VEGETATION HEALTH CLASSIFICATION
# ============================================================

def classify_vegetation(ndvi):
    """
    Classify vegetation health using NDVI values.

    Classes:

    0 = Water / Very Low Vegetation
    1 = Bare Soil / Non-Vegetated
    2 = Sparse / Stressed Vegetation
    3 = Moderate Vegetation
    4 = Healthy Vegetation
    """

    health_map = np.zeros(
        ndvi.shape,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Class 0: Water / very low vegetation
    # NDVI < 0
    # --------------------------------------------------------

    health_map[
        ndvi < 0
    ] = 0

    # --------------------------------------------------------
    # Class 1: Bare soil / non-vegetated
    # 0.0 <= NDVI < 0.2
    # --------------------------------------------------------

    health_map[
        (ndvi >= 0.0) &
        (ndvi < 0.2)
    ] = 1

    # --------------------------------------------------------
    # Class 2: Sparse / stressed vegetation
    # 0.2 <= NDVI < 0.4
    # --------------------------------------------------------

    health_map[
        (ndvi >= 0.2) &
        (ndvi < 0.4)
    ] = 2

    # --------------------------------------------------------
    # Class 3: Moderate vegetation
    # 0.4 <= NDVI < 0.6
    # --------------------------------------------------------

    health_map[
        (ndvi >= 0.4) &
        (ndvi < 0.6)
    ] = 3

    # --------------------------------------------------------
    # Class 4: Healthy vegetation
    # NDVI >= 0.6
    # --------------------------------------------------------

    health_map[
        ndvi >= 0.6
    ] = 4

    return health_map


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Water / Very Low Vegetation",
    1: "Bare Soil / Non-Vegetated",
    2: "Sparse / Stressed Vegetation",
    3: "Moderate Vegetation",
    4: "Healthy Vegetation"
}


# ============================================================
# FIND NDVI FILES
# ============================================================

print("=" * 70)
print("VEGETATION HEALTH ANALYSIS")
print("=" * 70)


ndvi_files = sorted([
    f
    for f in os.listdir(NDVI_DIR)
    if f.endswith("_ndvi.npy")
])


ndvi_files = ndvi_files[:MAX_SAMPLES]


print(
    "\nNumber of NDVI images:",
    len(ndvi_files)
)


if len(ndvi_files) == 0:

    print(
        "\nNo NDVI files found."
    )

    print(
        "Please run ndvi_analysis.py first."
    )

    raise SystemExit


# ============================================================
# STATISTICS STORAGE
# ============================================================

total_pixels = 0

class_pixel_counts = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0
}

mean_ndvi_values = []


# ============================================================
# PROCESS EACH NDVI IMAGE
# ============================================================

for index, filename in enumerate(ndvi_files):

    print(
        f"\nProcessing "
        f"{index + 1}/{len(ndvi_files)}: "
        f"{filename}"
    )

    # --------------------------------------------------------
    # Load NDVI
    # --------------------------------------------------------

    ndvi_path = os.path.join(
        NDVI_DIR,
        filename
    )

    ndvi = np.load(
        ndvi_path
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    ndvi = np.nan_to_num(
        ndvi,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0
    )

    ndvi = np.clip(
        ndvi,
        -1,
        1
    )

    # --------------------------------------------------------
    # Calculate mean NDVI
    # --------------------------------------------------------

    mean_ndvi = float(
        np.mean(ndvi)
    )

    mean_ndvi_values.append(
        mean_ndvi
    )

    # --------------------------------------------------------
    # Classify vegetation
    # --------------------------------------------------------

    health_map = classify_vegetation(
        ndvi
    )

    # --------------------------------------------------------
    # Count pixels
    # --------------------------------------------------------

    total_pixels += health_map.size

    for class_id in range(5):

        count = np.sum(
            health_map == class_id
        )

        class_pixel_counts[
            class_id
        ] += int(count)

    # --------------------------------------------------------
    # Save vegetation health map
    # --------------------------------------------------------

    base_name = filename.replace(
        "_ndvi.npy",
        ""
    )

    output_npy = os.path.join(
        OUTPUT_DIR,
        base_name + "_vegetation_health.npy"
    )

    np.save(
        output_npy,
        health_map
    )

    # --------------------------------------------------------
    # Print image statistics
    # --------------------------------------------------------

    print(
        f"Mean NDVI: {mean_ndvi:.4f}"
    )

    for class_id in range(5):

        count = np.sum(
            health_map == class_id
        )

        percentage = (
            count /
            health_map.size
        ) * 100

        print(
            f"  {CLASS_NAMES[class_id]}: "
            f"{percentage:.2f}%"
        )


# ============================================================
# OVERALL CLASS PERCENTAGES
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL VEGETATION HEALTH RESULTS")
print("=" * 70)


class_percentages = {}


for class_id in range(5):

    percentage = (
        class_pixel_counts[class_id]
        /
        total_pixels
    ) * 100

    class_percentages[
        class_id
    ] = percentage

    print(
        f"\n{CLASS_NAMES[class_id]}"
    )

    print(
        f"Percentage: {percentage:.2f}%"
    )


# ============================================================
# OVERALL MEAN NDVI
# ============================================================

overall_mean_ndvi = np.mean(
    mean_ndvi_values
)


print(
    "\nAverage Mean NDVI:",
    f"{overall_mean_ndvi:.4f}"
)


# ============================================================
# DETERMINE OVERALL VEGETATION CONDITION
# ============================================================

healthy_percentage = class_percentages[4]

moderate_percentage = class_percentages[3]

stressed_percentage = class_percentages[2]

bare_percentage = class_percentages[1]

water_percentage = class_percentages[0]


if healthy_percentage >= 50:

    overall_condition = (
        "The area contains predominantly healthy vegetation."
    )

elif (
    healthy_percentage +
    moderate_percentage
) >= 50:

    overall_condition = (
        "The area contains moderate to healthy vegetation."
    )

elif stressed_percentage >= 20:

    overall_condition = (
        "The area contains a significant amount "
        "of sparse or stressed vegetation."
    )

else:

    overall_condition = (
        "The area contains relatively low vegetation "
        "coverage."
    )


print(
    "\nOverall condition:"
)

print(
    overall_condition
)


# ============================================================
# CREATE VISUALIZATION FOR FIRST IMAGE
# ============================================================

first_file = ndvi_files[0]

first_path = os.path.join(
    NDVI_DIR,
    first_file
)

first_ndvi = np.load(
    first_path
).astype(
    np.float32
)

first_ndvi = np.nan_to_num(
    first_ndvi,
    nan=0.0,
    posinf=1.0,
    neginf=-1.0
)

first_ndvi = np.clip(
    first_ndvi,
    -1,
    1
)


first_health_map = classify_vegetation(
    first_ndvi
)


# ============================================================
# SAVE HEALTH MAP
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    first_health_map,
    cmap="RdYlGn",
    vmin=0,
    vmax=4
)

cbar = plt.colorbar()

cbar.set_ticks(
    [0, 1, 2, 3, 4]
)

cbar.set_ticklabels([
    "Water / Very Low",
    "Bare / Non-Vegetated",
    "Sparse / Stressed",
    "Moderate",
    "Healthy"
])

plt.title(
    "Vegetation Health Map"
)

plt.axis(
    "off"
)

plt.tight_layout()


health_map_path = os.path.join(
    OUTPUT_DIR,
    "vegetation_health_map.png"
)


plt.savefig(
    health_map_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "vegetation_health_report.txt"
)


with open(
    report_path,
    "w"
) as f:

    f.write(
        "VEGETATION HEALTH ANALYSIS REPORT\n"
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
        "Analysis: NDVI-based Vegetation Health\n"
    )

    f.write(
        f"Number of Images: {len(ndvi_files)}\n\n"
    )

    f.write(
        "NDVI HEALTH CLASSES\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        "Class 0: NDVI < 0 "
        "-> Water / Very Low Vegetation\n"
    )

    f.write(
        "Class 1: 0.0 <= NDVI < 0.2 "
        "-> Bare Soil / Non-Vegetated\n"
    )

    f.write(
        "Class 2: 0.2 <= NDVI < 0.4 "
        "-> Sparse / Stressed Vegetation\n"
    )

    f.write(
        "Class 3: 0.4 <= NDVI < 0.6 "
        "-> Moderate Vegetation\n"
    )

    f.write(
        "Class 4: NDVI >= 0.6 "
        "-> Healthy Vegetation\n\n"
    )

    f.write(
        "OVERALL RESULTS\n"
    )

    f.write(
        "-" * 60 + "\n"
    )

    f.write(
        f"Average Mean NDVI: "
        f"{overall_mean_ndvi:.4f}\n\n"
    )

    for class_id in range(5):

        f.write(
            f"{CLASS_NAMES[class_id]}: "
            f"{class_percentages[class_id]:.2f}%\n"
        )

    f.write(
        "\nOverall Condition:\n"
    )

    f.write(
        overall_condition
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n")
print("=" * 70)
print("VEGETATION HEALTH ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nVegetation health map saved to:"
)

print(
    health_map_path
)

print(
    "\nVegetation health report saved to:"
)

print(
    report_path
)

print(
    "\nVegetation health arrays saved in:"
)

print(
    OUTPUT_DIR
)