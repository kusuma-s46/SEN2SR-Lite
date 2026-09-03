import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels=64, res_scale=0.1):
        super().__init__()

        self.res_scale = res_scale

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):

        residual = self.block(x)

        return x + residual * self.res_scale


class UpsampleBlock(nn.Module):

    def __init__(self, channels=64, scale=4):

        super().__init__()

        layers = []

        # For ×4, perform ×2 twice
        for _ in range(scale // 2):

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
                nn.ReLU(inplace=True)
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)


class EDSR(nn.Module):

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        num_features=64,
        num_blocks=16,
        scale=4
    ):

        super().__init__()

        # First feature extraction layer
        self.head = nn.Conv2d(
            in_channels,
            num_features,
            kernel_size=3,
            padding=1
        )

        # Residual blocks
        residual_blocks = []

        for _ in range(num_blocks):

            residual_blocks.append(
                ResidualBlock(
                    channels=num_features
                )
            )

        # Body
        self.body = nn.Sequential(
            *residual_blocks,

            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                padding=1
            )
        )

        # Upsampling
        self.upsample = UpsampleBlock(
            channels=num_features,
            scale=scale
        )

        # Final reconstruction layer
        self.tail = nn.Conv2d(
            num_features,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # Head
        features = self.head(x)

        # Residual body
        body = self.body(features)

        # Global residual connection
        body = body + features

        # Upsampling
        upsampled = self.upsample(body)

        # Final image
        output = self.tail(upsampled)

        return output


# --------------------------------------------------
# Test EDSR
# --------------------------------------------------

if __name__ == "__main__":

    print("Testing EDSR model...")

    # Create model
    model = EDSR(
        in_channels=4,
        out_channels=4,
        num_features=64,
        num_blocks=16,
        scale=4
    )

    print("\nModel created successfully!")

    # Create fake Sentinel-2 LR input
    input_image = torch.randn(
        1,
        4,
        130,
        130
    )

    print("Input shape:", input_image.shape)

    # Forward pass
    output = model(input_image)

    print("Output shape:", output.shape)

    # Number of parameters
    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Total parameters:", total_parameters)

    print("\nEDSR test completed successfully!")