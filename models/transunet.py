import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, image_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: (batch_size, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.pos_embed


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.msa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.msa(x, x, x)
        return attn_output


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.msa = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # MSA block
        x_msa = self.msa(self.norm1(x))
        x = x + self.dropout(x_msa)

        # MLP block
        x_mlp = self.mlp(self.norm2(x))
        x = x + self.dropout(x_mlp)

        return x


class ViTEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, num_layers, patch_size, image_size, mlp_dim):
        super(ViTEncoder, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size, image_size)
        self.pos_embed = PositionalEncoding(embed_dim, self.patch_embed.n_patches)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upconv(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
        
class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)  # Downsampling by 2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x
        
class ViT_UNet(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=512, num_heads=4, num_layers=4, patch_size=16, image_size=256, mlp_dim=1024):
        super(ViT_UNet, self).__init__()
        
        # Add convolutional encoder blocks before ViT
        self.conv_encoder1 = ConvEncoderBlock(in_channels, 64)  # First conv block
        self.conv_encoder2 = ConvEncoderBlock(64, 128)  # Second conv block
                
        # ViT Encoder
        self.encoder = ViTEncoder(128, embed_dim, num_heads, num_layers, patch_size, image_size // 4, mlp_dim)
        
        # Upsample layer for ViT output to match encoder's size
        self.vit_upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        
        # Decoder blocks
        self.decoder1 = DecoderBlock(embed_dim + 128, 128)  # Concatenate ViT output and conv_encoder2 output
        self.decoder2 = DecoderBlock(128, 16)

        #self.decoder3 = DecoderBlock(64, 16)  # Extra decoder block
        #self.decoder4 = DecoderBlock(32, 16)  # Another extra decoder block
        
        # Final upsampling and classification layer
        #self.upsample_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Convolutional encoder blocks
        enc1 = self.conv_encoder1(x)
        enc2 = self.conv_encoder2(enc1)
        
        # ViT encoder
        vit_out = self.encoder(enc2)
        
        # Reshape and upsample ViT output to match enc2 spatial dimensions
        vit_out = vit_out.transpose(1, 2)
        vit_out = vit_out.view(vit_out.size(0), -1, int(enc2.size(2) // 16), int(enc2.size(3) // 16))
        vit_out = self.vit_upsample(vit_out)  # Upsample to match enc2 size
        
        # Decoder blocks with skip connections
        dec1 = self.decoder1(torch.cat([vit_out, enc2], dim=1))  # Skip connection with conv_encoder2 output
        dec2 = self.decoder2(dec1)
#        #dec4 = self.decoder4(dec3)  # Additional decoder block
        
        # Upsample and final output
        #dec4_upsampled = self.upsample_final(dec4)
        out = self.final_conv(dec2)
        
        return out



#class ViT_UNet(nn.Module):
#    def __init__(self, in_channels, num_classes, embed_dim=512, num_heads=4, num_layers=6, patch_size=16, image_size=256, mlp_dim=1024):
#        super(ViT_UNet, self).__init__()
#        
#        # Convolutional encoder blocks
#        self.conv_encoder1 = ConvEncoderBlock(in_channels, 64)  # First conv block
#        self.conv_encoder2 = ConvEncoderBlock(64, 128)  # Second conv block
#        
#        # ViT Encoder
#        self.encoder = ViTEncoder(128, embed_dim, num_heads, num_layers, patch_size, image_size // 4, mlp_dim)
#        
#        # Decoder blocks
#        self.decoder1 = DecoderBlock(embed_dim, 128)
#        self.decoder2 = DecoderBlock(128, 64)
#        self.decoder3 = DecoderBlock(64, 32)
#        self.decoder4 = DecoderBlock(32, 16)
#        
#        # Final upsampling and classification layer
#        self.upsample_final = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
#    
#    def forward(self, x):
#        # Convolutional encoder blocks
#        enc1 = self.conv_encoder1(x)  # Shape: (B, 64, H/2, W/2)
#        enc2 = self.conv_encoder2(enc1)  # Shape: (B, 128, H/4, W/4)
#        
#        # ViT encoder
#        enc_out = self.encoder(enc2)  # Shape: (B, num_patches, embed_dim)
#        
#        # Reshape ViT output to match the decoder input shape
#        enc_out = enc_out.transpose(1, 2)  # Shape: (B, embed_dim, num_patches)
#        enc_out = enc_out.view(enc_out.size(0), -1, int(enc2.size(2) // 4), int(enc2.size(3) // 4))  # Shape: (B, embed_dim, H/16, W/16)
#        
#        # Decoder blocks with skip connections
#        dec1 = self.decoder1(enc_out)  # Shape: (B, 128, H/8, W/8)
#        dec2 = self.decoder2(dec1)  # Shape: (B, 64, H/4, W/4)
#        dec2 = torch.cat([dec2, enc2], dim=1)  # Concatenate with encoder output (skip connection)
#        
#        dec3 = self.decoder3(dec2)  # Shape: (B, 32, H/2, W/2)
#        dec3 = torch.cat([dec3, enc1], dim=1)  # Concatenate with encoder output (skip connection)
#        
#        dec4 = self.decoder4(dec3)  # Shape: (B, 16, H, W)
#        
#        # Upsample and final output
#        dec4_upsampled = self.upsample_final(dec4)
#        out = self.final_conv(dec4_upsampled)
#        
#        return out
#
#
#
##class ViT_UNet(nn.Module):
##    def __init__(self, in_channels, num_classes, embed_dim=512, num_heads=4, num_layers=6, patch_size=16, image_size=256, mlp_dim=1024):
##        super(ViT_UNet, self).__init__()
##        
##        # Add convolutional encoder blocks before ViT
##        self.conv_encoder1 = ConvEncoderBlock(in_channels, 64)  # First conv block
##        self.conv_encoder2 = ConvEncoderBlock(64, 128)  # Second conv block
##        
##        # ViT Encoder
##        self.encoder = ViTEncoder(128, embed_dim, num_heads, num_layers, patch_size, image_size // 4, mlp_dim)
##        
##        # Decoder blocks
##        self.decoder1 = DecoderBlock(embed_dim, 128)
##        self.decoder2 = DecoderBlock(128, 64)
##        self.decoder3 = DecoderBlock(64, 32)  # Extra decoder block
##        self.decoder4 = DecoderBlock(32, 16)  # Another extra decoder block
##        
##        # Final upsampling and classification layer
##        self.upsample_final = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
##        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
##    
##    def forward(self, x):
##        # Convolutional encoder blocks
##        x = self.conv_encoder1(x)
##        x = self.conv_encoder2(x)
##        
##        # ViT encoder
##        enc_out = self.encoder(x)
##        
##        # Reshape and prepare for decoder
##        enc_out = enc_out.transpose(1, 2)
##        enc_out = enc_out.view(enc_out.size(0), -1, int(x.size(2) // 16), int(x.size(3) // 16))
##        
##        # Decoder blocks
##        dec1 = self.decoder1(enc_out)
##        dec2 = self.decoder2(dec1)
##        dec3 = self.decoder3(dec2)  # Additional decoder block
##        dec4 = self.decoder4(dec3)  # Additional decoder block
##        
##        # Upsample and final output
##        dec4_upsampled = self.upsample_final(dec4)
##        out = self.final_conv(dec4_upsampled)
##        
##        return out
#
#
##class ViT_UNet(nn.Module):
##    def __init__(self, in_channels, num_classes, embed_dim=512, num_heads=8, num_layers=6, patch_size=16, image_size=256, mlp_dim=1024):
##        super(ViT_UNet, self).__init__()
##        
##        # Encoder using ViT
##        self.encoder = ViTEncoder(in_channels, embed_dim, num_heads, num_layers, patch_size, image_size, mlp_dim)
##        
##        # Decoder
##        self.decoder1 = DecoderBlock(embed_dim, 128)
##        self.decoder2 = DecoderBlock(128, 64)
##        
##        # Additional upsampling to ensure output is [B, num_classes, 256, 256]
##        self.upsample_final = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
##        
##        # Final convolution to output the number of classes
##        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
##    
##    def forward(self, x):
##        # Encoder
##        enc_out = self.encoder(x)
##        
##        # Decoder (reshape and process encoded output)
##        enc_out = enc_out.transpose(1, 2)  # Reshape to (batch_size, embed_dim, num_patches)
##        enc_out = enc_out.view(enc_out.size(0), -1, int(x.size(2) // 16), int(x.size(3) // 16))
##        
##        dec1 = self.decoder1(enc_out)  # Output shape: [batch_size, 128, 32, 32]
##        dec2 = self.decoder2(dec1)  # Output shape: [batch_size, 64, 64, 64]
##        
##        # Upsample to [batch_size, 64, 256, 256] before final classification layer
##        dec2_upsampled = self.upsample_final(dec2)
##        
##        # Final output: [batch_size, num_classes, 256, 256]
##        out = self.final_conv(dec2_upsampled)
##        return out


## Example usage
#if __name__ == "__main__":
##     Change input_channels to 15 for your case
#    model = ViT_UNet(in_channels=16, num_classes=9)
#    input_tensor = torch.randn(1, 16, 256, 256)  # Batch size 1, 15 channels, 256x256 image
#    output = model(input_tensor)
#    print(output.shape)  # Expected output: (1, 9, 256, 256)
 