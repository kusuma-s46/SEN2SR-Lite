import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic Residual Block
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.body(x)


# ============================================================
# Window Partition
# ============================================================

def window_partition(x, window_size):
    """
    Convert feature map into non-overlapping windows.

    Input:
        x: [B, H, W, C]

    Output:
        windows: [num_windows * B, window_size, window_size, C]
    """

    B, H, W, C = x.shape

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )

    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
         .contiguous()
         .view(-1, window_size, window_size, C)
    )

    return windows


# ============================================================
# Window Reverse
# ============================================================

def window_reverse(windows, window_size, H, W):
    """
    Reconstruct feature map from windows.

    Input:
        windows: [num_windows * B, window_size, window_size, C]

    Output:
        x: [B, H, W, C]
    """

    B = int(
        windows.shape[0] /
        ((H // window_size) * (W // window_size))
    )

    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1
    )

    x = (
        x.permute(0, 1, 3, 2, 4, 5)
         .contiguous()
         .view(B, H, W, -1)
    )

    return x


# ============================================================
# Window based Self Attention
# ============================================================

class WindowAttention(nn.Module):

    def __init__(
        self,
        dim,
        window_size,
        num_heads
    ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):

        B_, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                B_,
                N,
                3,
                self.num_heads,
                self.head_dim
            )
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attention = q @ k.transpose(-2, -1)

        attention = F.softmax(
            attention,
            dim=-1
        )

        x = (
            attention @ v
        )

        x = (
            x.transpose(1, 2)
             .reshape(B_, N, C)
        )

        x = self.proj(x)

        return x


# ============================================================
# Swin Transformer Block
# ============================================================

class SwinTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        shift_size=0,
        mlp_ratio=2.0
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)

        self.attention = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(
            dim * mlp_ratio
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):

        B, H, W, C = x.shape

        shortcut = x

        x = self.norm1(x)

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        pad_h = (
            self.window_size -
            H % self.window_size
        ) % self.window_size

        pad_w = (
            self.window_size -
            W % self.window_size
        ) % self.window_size

        if pad_h > 0 or pad_w > 0:

            x = x.permute(
                0, 3, 1, 2
            )

            x = F.pad(
                x,
                (0, pad_w, 0, pad_h)
            )

            x = x.permute(
                0, 2, 3, 1
            )

        Hp, Wp = x.shape[1], x.shape[2]

        # ----------------------------------------------------
        # Shifted Window
        # ----------------------------------------------------

        if self.shift_size > 0:

            shifted_x = torch.roll(
                x,
                shifts=(
                    -self.shift_size,
                    -self.shift_size
                ),
                dims=(1, 2)
            )

        else:

            shifted_x = x

        # ----------------------------------------------------
        # Window Partition
        # ----------------------------------------------------

        windows = window_partition(
            shifted_x,
            self.window_size
        )

        windows = windows.view(
            -1,
            self.window_size *
            self.window_size,
            C
        )

        # ----------------------------------------------------
        # Window Attention
        # ----------------------------------------------------

        attention_windows = self.attention(
            windows
        )

        # ----------------------------------------------------
        # Reverse Windows
        # ----------------------------------------------------

        attention_windows = attention_windows.view(
            -1,
            self.window_size,
            self.window_size,
            C
        )

        shifted_x = window_reverse(
            attention_windows,
            self.window_size,
            Hp,
            Wp
        )

        # ----------------------------------------------------
        # Reverse Shift
        # ----------------------------------------------------

        if self.shift_size > 0:

            x = torch.roll(
                shifted_x,
                shifts=(
                    self.shift_size,
                    self.shift_size
                ),
                dims=(1, 2)
            )

        else:

            x = shifted_x

        # ----------------------------------------------------
        # Remove Padding
        # ----------------------------------------------------

        if pad_h > 0 or pad_w > 0:

            x = x[:, :H, :W, :]

        # ----------------------------------------------------
        # Residual Connection
        # ----------------------------------------------------

        x = shortcut + x

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        x = (
            x +
            self.mlp(
                self.norm2(x)
            )
        )

        return x


# ============================================================
# Swin Transformer Layer
# ============================================================

class BasicLayer(nn.Module):

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=8
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(depth):

            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(
                        0
                        if i % 2 == 0
                        else window_size // 2
                    )
                )
            )

    def forward(self, x):

        for block in self.blocks:

            x = block(x)

        return x


# ============================================================
# SwinIR Multispectral Model
# ============================================================

class SwinIR(nn.Module):

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        embed_dim=64,
        depths=4,
        num_heads=4,
        window_size=8,
        upscale=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale = upscale

        # ----------------------------------------------------
        # Shallow Feature Extraction
        # ----------------------------------------------------

        self.conv_first = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # ----------------------------------------------------
        # Swin Transformer Body
        # ----------------------------------------------------

        self.layers = BasicLayer(
            dim=embed_dim,
            depth=depths,
            num_heads=num_heads,
            window_size=window_size
        )

        # ----------------------------------------------------
        # Feature Reconstruction
        # ----------------------------------------------------

        self.conv_after_body = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                embed_dim,
                embed_dim * upscale * upscale,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.PixelShuffle(upscale),

            nn.GELU(),

            nn.Conv2d(
                embed_dim,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        shallow_features = self.conv_first(x)

        # ----------------------------------------------------
        # Convert BCHW → BHWC
        # ----------------------------------------------------

        features = shallow_features.permute(
            0,
            2,
            3,
            1
        )

        # ----------------------------------------------------
        # Swin Transformer
        # ----------------------------------------------------

        features = self.layers(
            features
        )

        # ----------------------------------------------------
        # Convert BHWC → BCHW
        # ----------------------------------------------------

        features = features.permute(
            0,
            3,
            1,
            2
        )

        # ----------------------------------------------------
        # Deep Feature Reconstruction
        # ----------------------------------------------------

        body_features = self.conv_after_body(
            features
        )

        # ----------------------------------------------------
        # Residual Connection
        # ----------------------------------------------------

        body_features = (
            body_features +
            shallow_features
        )

        # ----------------------------------------------------
        # 4× Super Resolution
        # ----------------------------------------------------

        output = self.upsample(
            body_features
        )

        return output


# ============================================================
# Model Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SwinIR Multispectral Super-Resolution Model Test")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Create Model
    # --------------------------------------------------------

    model = SwinIR(
        in_channels=4,
        out_channels=4,
        embed_dim=64,
        depths=4,
        num_heads=4,
        window_size=8,
        upscale=4
    ).to(device)

    # --------------------------------------------------------
    # Print Model
    # --------------------------------------------------------

    print("\nModel created successfully.")

    # --------------------------------------------------------
    # Dummy Sentinel-2 LR Input
    # --------------------------------------------------------

    x = torch.randn(
        1,
        4,
        130,
        130
    ).to(device)

    print("\nInput shape:")
    print(x.shape)

    # --------------------------------------------------------
    # Forward Pass
    # --------------------------------------------------------

    with torch.no_grad():

        y = model(x)

    print("\nOutput shape:")
    print(y.shape)

    # --------------------------------------------------------
    # Expected Output
    # --------------------------------------------------------

    print("\nExpected:")
    print("(1, 4, 520, 520)")

    # --------------------------------------------------------
    # Parameter Count
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTotal parameters:")
    print(f"{total_params:,}")

    print("\nTrainable parameters:")
    print(f"{trainable_params:,}")

    print("\n" + "=" * 60)
    print("SwinIR model test completed successfully.")
    print("=" * 60)