import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderBlock(nn.Module):
    def __init__(self, resnet_model, freeze=False):
        super().__init__()
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu1 = resnet_model.relu
        self.mp1 = resnet_model.maxpool

        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()



    def forward(self, x):
        x1 = self.relu1(self.bn1(self.conv1(x))) # x -> x1: (B, 64, 128, 128)
        x2 = self.layer1(self.mp1(x1)) # x1 -> x2: (B, 64, 64, 64)
        x3 = self.layer2(x2) # x2 -> x3: (B, 128, 32, 32)
        x4 = self.layer3(x3) # x3 -> x4: (B, 256, 16, 16)
        x5 = self.layer4(x4) # x4 -> x5: (B, 512, 8, 8)
        return x1, x2, x3, x4, x5
    
    

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # x = self.upsample(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, skip), dim=1)  # Concatenate along channel dimension
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x
    
class Decoder(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.up4 = UpSampleBlock(512 + 256, 256)
        self.up3 = UpSampleBlock(256 + 128, 128)
        self.up2 = UpSampleBlock(128 + 64, 64)
        self.up1 = UpSampleBlock(64 + 64, 64)

        # self.final_upsample = nn.Sequential(
        #     [
        #         nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        #     ]
        # )
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # Assuming binary segmentation

    def forward(self, x1, x2, x3, x4, x5):
        """
        x1: (B, 64, 128, 128)
        x2: (B, 64, 64, 64)
        x3: (B, 128, 32, 32)
        x4: (B, 256, 16, 16)
        x5: (B, 512, 8, 8)
        """
        d4 = self.up4(x5, x4)  # x5 and x4: d4: (B, 256, 16, 16)
        d3 = self.up3(d4, x3)  # d4 and x3: d3: (B, 128, 32, 32)
        d2 = self.up2(d3, x2)  # d3 and x2: d2: (B, 64, 64, 64)
        d1 = self.up1(d2, x1)  # d2 and x1: d1: (B, 64, 128, 128)
        # add d0 for a final up sample before final conv if needed.
        out = self.final_conv(d1)
        return out
    

class ResNet34UNet(nn.Module):
    def __init__(self, resnet_model=models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1),
                 freeze_encoder=False, num_classes=21, **kwargs):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder = EncoderBlock(resnet_model, freeze=self.freeze_encoder)
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        out = self.decoder(x1, x2, x3, x4, x5)
        return out
    
    def train(self, mode: bool = True):
        """
        Override the default train() to freeze the encoder if specified. 
        Specifically, keep BatchNorm layers in eval mode.
        """
        super().train(mode)
        if self.freeze_encoder:
            # keep encoder BN in eval even when the rest is training
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

class SegmentationWrapper(nn.Module):
    """
    Wraps a base model to ensure output size matches input size via interpolation.
    Useful for segmentation models where input and output spatial dimensions must align.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        input_size = x.size()[2:]  # H, W
        x = self.base_model(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

        
if __name__ == "__main__":
    # simple test
    print("unet.py test: \n Loading Model...")
    model = ResNet34UNet()
    model = SegmentationWrapper(model)
    rand_im = torch.randn((2, 3, 256, 256))  # batch of 2, 3 channels, 256x256
    rand_im2 = torch.randn((2, 3, 312, 192))  # batch of 2, 3 channels, 256x256
    print("Running forward pass...")
    out = model(rand_im)
    print("out.shape: ", out.shape)  # expect (2, 1, 256, 256)
    s = torch.sum(out)
    s.backward()
    out2 = model(rand_im2)
    print("out2.shape: ", out2.shape)  # expect (2, 1, 312, 192)


    print("Done!")
    print("unet.py test with frozen weights: \n Loading Model...")
    model = ResNet34UNet(freeze_encoder=True)
    rand_im = torch.randn((2, 3, 256, 256))  # batch of 2, 3 channels, 256x256
    print("Running forward pass...")
    out = model(rand_im)
    print(out.shape)  # expect (2, 1, 256, 256)
    s = torch.sum(out)
    s.backward()
    print("Done!")
