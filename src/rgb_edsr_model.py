"""
RGB-EDSR Super-Resolution Model

Input:
    RGB Low-Resolution image

Output:
    RGB Super-Resolved image

Scale:
    4x

Example:
    130 x 130  ->  520 x 520
"""

import torch
import torch.nn as nn


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, features=64, res_scale=0.1):

        super().__init__()

        self.res_scale = res_scale

        self.block = nn.Sequential(

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        residual = self.block(x)

        return x + residual * self.res_scale


# ============================================================
# UPSAMPLING BLOCK
# ============================================================

class UpsampleBlock(nn.Module):

    def __init__(self, features=64, scale=4):

        super().__init__()

        layers = []

        # For 4x upscaling:
        # 2x + 2x = 4x

        if scale == 4:

            layers.append(
                nn.Conv2d(
                    features,
                    features * 4,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(
                nn.PixelShuffle(2)
            )

            layers.append(
                nn.ReLU(inplace=True)
            )

            layers.append(
                nn.Conv2d(
                    features,
                    features * 4,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(
                nn.PixelShuffle(2)
            )

            layers.append(
                nn.ReLU(inplace=True)
            )

        elif scale == 2:

            layers.append(
                nn.Conv2d(
                    features,
                    features * 4,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(
                nn.PixelShuffle(2)
            )

            layers.append(
                nn.ReLU(inplace=True)
            )

        else:

            raise ValueError(
                "Only scale=2 or scale=4 is supported."
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)


# ============================================================
# RGB EDSR
# ============================================================

class RGB_EDSR(nn.Module):

    def __init__(
        self,
        scale=4,
        num_features=64,
        num_blocks=16
    ):

        super().__init__()

        self.scale = scale

        # ----------------------------------------------------
        # Head
        # ----------------------------------------------------

        self.head = nn.Conv2d(
            3,
            num_features,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Body
        # ----------------------------------------------------

        body = []

        for _ in range(num_blocks):

            body.append(
                ResidualBlock(
                    features=num_features,
                    res_scale=0.1
                )
            )

        body.append(
            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                padding=1
            )
        )

        self.body = nn.Sequential(*body)

        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        self.upsample = UpsampleBlock(
            features=num_features,
            scale=scale
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        self.tail = nn.Conv2d(
            num_features,
            3,
            kernel_size=3,
            padding=1
        )

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    def forward(self, x):

        # Initial feature extraction

        head = self.head(x)

        # Residual feature learning

        body = self.body(head)

        # Global residual connection

        features = head + body

        # 4x upsampling

        upsampled = self.upsample(features)

        # RGB output

        output = self.tail(upsampled)

        return output


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RGB-EDSR MODEL TEST")
    print("=" * 60)

    # Create model

    model = RGB_EDSR(
        scale=4,
        num_features=64,
        num_blocks=16
    )

    print("\nModel created successfully.")

    # Create fake RGB LR image

    input_image = torch.randn(
        1,
        3,
        130,
        130
    )

    print(
        "\nInput shape:",
        input_image.shape
    )

    # Forward pass

    with torch.no_grad():

        output = model(input_image)

    print(
        "Output shape:",
        output.shape
    )

    # Parameter count

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "\nTotal parameters:",
        f"{parameters:,}"
    )

    print("\nExpected:")
    print("Input  : [1, 3, 130, 130]")
    print("Output : [1, 3, 520, 520]")

    print("\nRGB-EDSR test completed successfully.")
    print("=" * 60)