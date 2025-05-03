import torch

nn = torch.nn
F = nn.functional


class DoubleConv(nn.Module):
    """Represent a double convolution"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        residual: bool = False,
    ):
        """Initialize module"""

        super().__init__()

        mid_channels = mid_channels if mid_channels else out_channels

        self.residual = residual
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""

        if self.residual:
            x = x + self.model(x)
            x = F.gelu(x)
        else:
            x = self.model(x)
            x = F.mish(x)

        return x


class Down(nn.Module):
    """Represents a downsample"""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize module"""

        super().__init__()

        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""

        x = self.model(x)
        return x


class UpSkip(nn.Module):
    """Represents an upsample with a residual connection"""

    def __init__(self, in_channels, out_channels):
        """Initialize module"""

        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.model = nn.Sequential(
            DoubleConv(in_channels, in_channels),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""

        x = self.upsample(x)
        x = torch.cat((x, skip_x), dim=1)
        x = self.model(x)

        return x


class UNet(nn.Module):
    """Represents a U-Net"""

    def __init__(self):
        """Initialize module"""

        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.conv = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.bottleneck = nn.Sequential(
            Down(128, 256),
            DoubleConv(256, 256),
            DoubleConv(256, 256),
            DoubleConv(256, 128),
        )
        self.up1 = UpSkip(256, 64)
        self.up2 = UpSkip(128, 32)
        self.up3 = UpSkip(64, 32)
        self.unet_out = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""

        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.bottleneck(x3)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.unet_out(x)

        return x
