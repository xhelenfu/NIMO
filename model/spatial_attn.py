import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(LightweightUNet, self).__init__()

        # Encoder (Downsampling)
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder (Upsampling)
        self.up3 = self.upconv_block(256, 128)
        self.up2 = self.upconv_block(128, 64)
        self.up1 = self.upconv_block(64, 32)

        # Final output layer
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(enc3)

        # Decoder path
        up3 = self.up3(bottleneck)
        up3 = torch.cat([up3, enc3], dim=1)  # Skip connection
        up2 = self.up2(up3)
        up2 = torch.cat([up2, enc2], dim=1)  # Skip connection
        up1 = self.up1(up2)
        up1 = torch.cat([up1, enc1], dim=1)  # Skip connection

        # Final output
        out = self.out_conv(up1)
        return out


# Example usage:
# input_tensor = torch.randn(1, 1, 256, 256)  # Example batch of size 1 with 1 channel and 256x256 image
# model = LightweightUNet(in_channels=1, out_channels=1)
# output = model(input_tensor)
# print(output.shape)  # Expected output shape: torch.Size([1, 1, 256, 256])
