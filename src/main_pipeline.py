"""
SEN2SR-Lite
Integrated Deep Learning Based Super-Resolution Mapping Pipeline

Pipeline:
1. Super-resolution model evaluation
2. Visual comparison
3. High-resolution validation
4. Spectral consistency
5. Uncertainty estimation
6. NDVI analysis
7. Vegetation health analysis
8. Dry/barren area detection
9. Agricultural area detection
10. Agricultural feature extraction

Run:
    python src/main_pipeline.py
"""

import os
import sys
import subprocess
from datetime import datetime


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "SEN2SR-Lite"

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")


# ============================================================
# TERMINAL COLORS
# ============================================================

RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BLUE = "\033[94m"
BOLD = "\033[1m"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print("\n")
    print("=" * 70)
    print(f"{BOLD}{CYAN}{title}{RESET}")
    print("=" * 70)


def print_step(step_number, total_steps, description):
    print("\n")
    print("-" * 70)
    print(
        f"{BOLD}{BLUE}"
        f"STEP {step_number}/{total_steps}: {description}"
        f"{RESET}"
    )
    print("-" * 70)


def check_path(path, description):
    """
    Check whether an important file/folder exists.
    """
    if os.path.exists(path):
        print(f"{GREEN}[OK] {description}{RESET}")
        return True

    print(f"{RED}[MISSING] {description}{RESET}")
    print(f"        Path: {path}")
    return False


def run_script(script_name):
    """
    Execute one Python script from the src directory.
    """

    script_path = os.path.join(SRC_DIR, script_name)

    if not os.path.exists(script_path):
        print(
            f"{YELLOW}[SKIPPED] Script not found: "
            f"{script_name}{RESET}"
        )
        return False

    print(f"\n{CYAN}Running: {script_name}{RESET}")

    try:

        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            check=False
        )

        if result.returncode == 0:

            print(
                f"{GREEN}[SUCCESS] {script_name} completed.{RESET}"
            )

            return True

        else:

            print(
                f"{RED}[ERROR] {script_name} failed "
                f"with exit code {result.returncode}.{RESET}"
            )

            return False

    except Exception as error:

        print(
            f"{RED}[ERROR] Could not execute "
            f"{script_name}{RESET}"
        )

        print(f"Reason: {error}")

        return False


# ============================================================
# PROJECT CHECK
# ============================================================

def check_project():

    print_header("PROJECT ENVIRONMENT CHECK")

    print(f"Project: {PROJECT_NAME}")
    print(f"Root: {PROJECT_ROOT}")

    print("\nChecking important folders...\n")

    check_path(
        DATASET_DIR,
        "Dataset directory"
    )

    check_path(
        os.path.join(
            DATASET_DIR,
            "local",
            "train",
            "lr"
        ),
        "Training LR dataset"
    )

    check_path(
        os.path.join(
            DATASET_DIR,
            "local",
            "train",
            "hr"
        ),
        "Training HR dataset"
    )

    check_path(
        os.path.join(
            DATASET_DIR,
            "local",
            "val",
            "lr"
        ),
        "Validation LR dataset"
    )

    check_path(
        os.path.join(
            DATASET_DIR,
            "local",
            "val",
            "hr"
        ),
        "Validation HR dataset"
    )

    check_path(
        os.path.join(
            DATASET_DIR,
            "local",
            "test",
            "lr"
        ),
        "Test LR dataset"
    )

    check_path(
        os.path.join(
            DATASET_DIR,
            "local",
            "test",
            "hr"
        ),
        "Test HR dataset"
    )

    print("\nChecking trained models...\n")

    check_path(
        os.path.join(
            MODELS_DIR,
            "edsr_local.pth"
        ),
        "EDSR trained model"
    )

    check_path(
        os.path.join(
            MODELS_DIR,
            "swinir_local.pth"
        ),
        "SwinIR trained model"
    )

    check_path(
        os.path.join(
            MODELS_DIR,
            "esrgan_local.pth"
        ),
        "ESRGAN trained model"
    )

    # Create results directory if required

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print(
        f"\n{GREEN}[OK] Results directory ready.{RESET}"
    )


# ============================================================
# PIPELINE
# ============================================================

def main():

    start_time = datetime.now()

    total_steps = 9

    print("\n")
    print("=" * 70)
    print(f"{BOLD}{CYAN}")
    print("             SEN2SR-Lite")
    print("   INTEGRATED SUPER-RESOLUTION PIPELINE")
    print(f"{RESET}")
    print("=" * 70)

    print("\nDeep Learning Based Super Resolution Mapping")
    print("Sentinel-2 10 m → 2.5 m")
    print("Bands: B02, B03, B04, B08")
    print("Framework: PyTorch")

    print(
        f"\nStarted at: "
        f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # --------------------------------------------------------
    # PROJECT CHECK
    # --------------------------------------------------------

    check_project()

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print_step(
        1,
        total_steps,
        "Super-Resolution Model Evaluation"
    )

    run_script(
        "evaluate_all_models.py"
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print_step(
        2,
        total_steps,
        "Generate Super-Resolution Visual Comparison"
    )

    run_script(
        "visualize_sr_results.py"
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print_step(
        3,
        total_steps,
        "High-Resolution Reference Validation"
    )

    run_script(
        "high_resolution_validation.py"
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print_step(
        4,
        total_steps,
        "Spectral Consistency Analysis"
    )

    run_script(
        "spectral_consistency.py"
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print_step(
        5,
        total_steps,
        "Uncertainty Estimation"
    )

    run_script(
        "uncertainty_map.py"
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print_step(
        6,
        total_steps,
        "NDVI Analysis"
    )

    run_script(
        "ndvi_analysis.py"
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    print_step(
        7,
        total_steps,
        "Vegetation and Agricultural Analysis"
    )

    # Vegetation health
    run_script(
        "vegetation_health.py"
    )

    # Dry / barren detection
    run_script(
        "dry_barren_detection.py"
    )

    # Agricultural area detection
    run_script(
        "agricultural_area_detection.py"
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    print_step(
        8,
        total_steps,
        "Agricultural Feature Extraction"
    )

    run_script(
        "agricultural_feature_extraction.py"
    )

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    print_step(
        9,
        total_steps,
        "Final Pipeline Summary"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    end_time = datetime.now()

    elapsed = end_time - start_time

    print("\n")
    print("=" * 70)
    print(f"{BOLD}{GREEN}PIPELINE EXECUTION COMPLETED{RESET}")
    print("=" * 70)

    print("\nFinal Pipeline:")

    print(
        """
    Sentinel-2 / SEN2NAIPv2
              |
              v
       Preprocessing
              |
              v
        SEN2SR-Lite
              |
       +------+------+
       |      |      |
      EDSR  SwinIR ESRGAN
       |      |      |
       +------+------+
              |
              v
       Model Evaluation
              |
              v
       Selected SR Model
              |
              v
        2.5 m SR Image
              |
       +------+------+
       |             |
       v             v
  Uncertainty      NDVI
      Map            |
                     v
          Vegetation Analysis
                     |
        +------------+------------+
        |            |            |
        v            v            v
   Vegetation     Dry/Barren   Agricultural
     Health        Detection      Area
        |            |            |
        +------------+------------+
                     |
                     v
          Agricultural Features
                     |
                     v
       Crop Growth / Condition
       and Agricultural Statistics
    """
    )

    print("\nOutput directories:")

    output_directories = [
        "sr_visual_comparison",
        "high_resolution_validation",
        "spectral_consistency",
        "uncertainty",
        "ndvi",
        "vegetation_health",
        "dry_barren",
        "agricultural_area",
        "agricultural_features"
    ]

    for directory in output_directories:

        path = os.path.join(
            RESULTS_DIR,
            directory
        )

        if os.path.exists(path):

            print(
                f"{GREEN}  [OK] results/{directory}{RESET}"
            )

        else:

            print(
                f"{YELLOW}  [--] results/{directory}{RESET}"
            )

    print(
        f"\n{GREEN}Start time:{RESET} "
        f"{start_time.strftime('%H:%M:%S')}"
    )

    print(
        f"{GREEN}End time:{RESET}   "
        f"{end_time.strftime('%H:%M:%S')}"
    )

    print(
        f"{GREEN}Total time:{RESET} {elapsed}"
    )

    print("\n")
    print("=" * 70)

    print(
        f"{BOLD}{GREEN}"
        "SEN2SR-Lite pipeline is ready."
        f"{RESET}"
    )

    print("=" * 70)
    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            f"\n\n{YELLOW}"
            "Pipeline interrupted by user."
            f"{RESET}"
        )

        sys.exit(1)

    except Exception as error:

        print(
            f"\n\n{RED}"
            "Unexpected pipeline error:"
            f"{RESET}"
        )

        print(error)

        sys.exit(1)
        