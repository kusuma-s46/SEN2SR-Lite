# src/esrgan_model.py

import torch
import torch.nn as nn


# ============================================================
# RESIDUAL DENSE BLOCK
# ============================================================

class ResidualDenseBlock(nn.Module):

    def __init__(self, channels=64, growth_channels=32):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels + growth_channels,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            channels + growth_channels * 2,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            channels + growth_channels * 3,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv5 = nn.Conv2d(
            channels + growth_channels * 4,
            channels,
            kernel_size=3,
            padding=1
        )

        self.leaky_relu = nn.LeakyReLU(
            negative_slope=0.2,
            inplace=True
        )

        self.residual_scale = 0.2

    def forward(self, x):

        x1 = self.leaky_relu(
            self.conv1(x)
        )

        x2 = self.leaky_relu(
            self.conv2(
                torch.cat([x, x1], dim=1)
            )
        )

        x3 = self.leaky_relu(
            self.conv3(
                torch.cat([x, x1, x2], dim=1)
            )
        )

        x4 = self.leaky_relu(
            self.conv4(
                torch.cat([x, x1, x2, x3], dim=1)
            )
        )

        x5 = self.conv5(
            torch.cat(
                [x, x1, x2, x3, x4],
                dim=1
            )
        )

        return x + x5 * self.residual_scale


# ============================================================
# RESIDUAL IN RESIDUAL DENSE BLOCK
# ============================================================

class RRDB(nn.Module):

    def __init__(
        self,
        channels=64,
        growth_channels=32
    ):
        super().__init__()

        self.rdb1 = ResidualDenseBlock(
            channels,
            growth_channels
        )

        self.rdb2 = ResidualDenseBlock(
            channels,
            growth_channels
        )

        self.rdb3 = ResidualDenseBlock(
            channels,
            growth_channels
        )

        self.residual_scale = 0.2

    def forward(self, x):

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        return x + out * self.residual_scale


# ============================================================
# UPSAMPLING BLOCK
# ============================================================

class UpsampleBlock(nn.Module):

    def __init__(self, channels, scale):
        super().__init__()

        layers = []

        if scale == 4:

            # First x2 upsampling
            layers.append(
                nn.Conv2d(
                    channels,
                    channels * 4,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(
                nn.PixelShuffle(2)
            )

            layers.append(
                nn.LeakyReLU(
                    0.2,
                    inplace=True
                )
            )

            # Second x2 upsampling
            layers.append(
                nn.Conv2d(
                    channels,
                    channels * 4,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(
                nn.PixelShuffle(2)
            )

            layers.append(
                nn.LeakyReLU(
                    0.2,
                    inplace=True
                )
            )

        elif scale == 2:

            layers.append(
                nn.Conv2d(
                    channels,
                    channels * 4,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(
                nn.PixelShuffle(2)
            )

            layers.append(
                nn.LeakyReLU(
                    0.2,
                    inplace=True
                )
            )

        else:

            raise ValueError(
                "Only scale 2 and 4 are supported."
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)


# ============================================================
# ESRGAN-STYLE GENERATOR
# ============================================================

class ESRGAN(nn.Module):

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        num_features=64,
        num_blocks=6,
        growth_channels=32,
        scale=4
    ):
        super().__init__()

        self.scale = scale

        # ----------------------------------------------------
        # Initial feature extraction
        # ----------------------------------------------------

        self.conv_first = nn.Conv2d(
            in_channels,
            num_features,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # RRDB blocks
        # ----------------------------------------------------

        rrdb_blocks = []

        for _ in range(num_blocks):

            rrdb_blocks.append(
                RRDB(
                    channels=num_features,
                    growth_channels=growth_channels
                )
            )

        self.rrdb_trunk = nn.Sequential(
            *rrdb_blocks
        )

        # ----------------------------------------------------
        # Trunk convolution
        # ----------------------------------------------------

        self.trunk_conv = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        self.upsampling = UpsampleBlock(
            num_features,
            scale
        )

        # ----------------------------------------------------
        # High-resolution convolution
        # ----------------------------------------------------

        self.conv_hr = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Final output layer
        # ----------------------------------------------------

        self.conv_last = nn.Conv2d(
            num_features,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.leaky_relu = nn.LeakyReLU(
            0.2,
            inplace=True
        )

    def forward(self, x):

        # Initial features
        feat = self.conv_first(x)

        # RRDB trunk
        trunk = self.rrdb_trunk(feat)

        trunk = self.trunk_conv(trunk)

        # Global residual connection
        feat = feat + trunk

        # Upsampling
        out = self.upsampling(feat)

        # High-resolution processing
        out = self.leaky_relu(
            self.conv_hr(out)
        )

        # Final multispectral output
        out = self.conv_last(out)

        return out


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ESRGAN MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Create ESRGAN
    # --------------------------------------------------------

    model = ESRGAN(
        in_channels=4,
        out_channels=4,
        num_features=64,
        num_blocks=6,
        growth_channels=32,
        scale=4
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Dummy Sentinel-2 input
    # --------------------------------------------------------

    input_tensor = torch.randn(
        1,
        4,
        130,
        130
    ).to(device)

    print(
        f"\nInput shape: {input_tensor.shape}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

    print(
        f"Output shape: {output.shape}"
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    expected_shape = (
        1,
        4,
        520,
        520
    )

    if output.shape == expected_shape:

        print("\nSUCCESS!")

        print(
            "\nESRGAN successfully converts:"
        )

        print(
            "Input : 4 × 130 × 130"
        )

        print(
            "Output: 4 × 520 × 520"
        )

    else:

        print("\nERROR!")

        print(
            f"Expected: {expected_shape}"
        )

        print(
            f"Received: {output.shape}"
        )

    print("\n" + "=" * 70)
    print("ESRGAN MODEL TEST COMPLETED")
    print("=" * 70)