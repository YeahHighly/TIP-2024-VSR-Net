import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2d => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    """Encoder with downsampling"""
    def __init__(self, in_channels, features):
        super(Encoder, self).__init__()
        self.conv = DoubleConv(in_channels, features)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        skip = x  # Save for skip connection
        x = self.pool(x)
        return x, skip  # Return downsampled + skip connection

class Decoder(nn.Module):
    """Decoder with upsampling"""
    def __init__(self, in_channels, features):
        super(Decoder, self).__init__()
        self.conv = DoubleConv(in_channels + features, features)  

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)  
        x = torch.cat([x, skip], dim=1)  
        return self.conv(x)

class CMMBase(nn.Module):
    """Dual-Path U-Net for Image-Cluster Merging"""
    def __init__(self, i_channels=3, m_channels=1, base_features=64):
        super(CMMBase, self).__init__()

        # Image Encoder
        self.enc_img1 = Encoder(i_channels, base_features)
        self.enc_img2 = Encoder(base_features, base_features * 2)
        self.enc_img3 = Encoder(base_features * 2, base_features * 4)

        # Mappings Encoder
        self.enc_maps1 = Encoder(m_channels, base_features)
        self.enc_maps2 = Encoder(base_features, base_features * 2)
        self.enc_maps3 = Encoder(base_features * 2, base_features * 4)

        # Bottleneck Fusion
        self.bottleneck = DoubleConv(base_features * 8, base_features * 8)

        # Decoder
        self.dec3 = Decoder(base_features * 8, base_features * 4)
        self.dec2 = Decoder(base_features * 4, base_features * 2)
        self.dec1 = Decoder(base_features * 2, base_features)

        # Final output layer
        self.final_conv = nn.Conv2d(base_features, 1, kernel_size=1)

    def forward(self, image_patches, mappings_patches):
        """Forward pass"""
        # Image Encoder
        img1, img1_skip = self.enc_img1(image_patches)
        img2, img2_skip = self.enc_img2(img1)
        img3, img3_skip = self.enc_img3(img2)

        # Mappings Encoder
        maps1, maps1_skip = self.enc_maps1(mappings_patches)
        maps2, maps2_skip = self.enc_maps2(maps1)
        maps3, maps3_skip = self.enc_maps3(maps2)

        # Bottleneck Fusion
        fusion = torch.cat([img3, maps3], dim=1)
        fusion = self.bottleneck(fusion)

        # Decoder
        x = self.dec3(fusion, img3_skip)
        x = self.dec2(x, img2_skip)
        x = self.dec1(x, img1_skip)

        # Final Output
        out = self.final_conv(x)
        return torch.sigmoid(out)  # Normalize output

# # Model Test
# model = CMMBase(base_features=64)

# # Dummy Inputs
# image_patches = torch.randn(4, 3, 256, 256)  # Batch of 4 images
# cluster_patches = torch.randn(4, 1, 256, 256)  # Batch of 4 cluster masks

# # Forward Pass
# output = model(image_patches, cluster_patches)

# print(f"Output Shape: {output.shape}")  # Expected: [4, 1, 256, 256]
