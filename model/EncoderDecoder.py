import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.DoubleAttention import Double_window_Double_attn

## self superised Module
class DSM(nn.Module):
    def __init__(self, in_channel, scale):
        super(DSM, self).__init__()
        self.refine = nn.Conv2d(in_channel, in_channel, 1, bias=False)
        self.predict = nn.Sequential(
                    nn.Conv2d(in_channel, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 1, 1, bias=False),
                    nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
                    )
        self.attn = nn.Sequential(
                        nn.Upsample(scale_factor=1/scale, mode='bilinear', align_corners=False)
                    )

    def forward(self, x):
        x1 = self.refine(x)
        shad = self.predict(x)
        attn = torch.sigmoid(self.attn(shad))
        x1 = x1*attn
        x = x+x1
        return shad, x


class EncoderDecoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.ca_lf = DSM(128, 4)

        self.attn1 = Double_window_Double_attn(256, (48, 48), 4, 24, 12)
        self.attn2 = Double_window_Double_attn(512, (24, 24), 8, 24, 12)
        self.attn3 = Double_window_Double_attn(1024, (12, 12), 16, 24, 12)
        self.attn4 = Double_window_Double_attn(1024, (12, 12), 32, 24, 12)

        self.attn12 = Double_window_Double_attn(256, (48, 48), 4, 24, 12)
        self.attn22 = Double_window_Double_attn(512, (24, 24), 8, 24, 12)
        self.attn32 = Double_window_Double_attn(1024, (12, 12), 16, 24, 12)
        self.attn42 = Double_window_Double_attn(1024, (12, 12), 32, 24, 12)


        self.conv_p5    = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),)
        self.conv_p5_p4 = nn.Sequential(nn.Conv2d(1024+1024, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),)
        self.conv_p4_p3 = nn.Sequential(nn.Conv2d(1024+512, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),)
        self.conv_p3_p2 = nn.Sequential(nn.Conv2d(512+256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),)
        self.conv_p2_p1 = nn.Sequential(nn.Conv2d(256+128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                                        nn.Conv2d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),)


        self.conv1x1_ReLU_lowft = nn.Sequential(nn.Conv2d(256*5, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                                nn.Conv2d(32, 1, 1, bias=False))
        self.conv1x1_ReLU_down1 = nn.Sequential(nn.Conv2d(256*4, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                                nn.Conv2d(32, 1, 1, bias=False))
        self.conv1x1_ReLU_down2 = nn.Sequential(nn.Conv2d(256*3, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                                nn.Conv2d(32, 1, 1, bias=False))
        self.conv1x1_ReLU_down3 = nn.Sequential(nn.Conv2d(256*2, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                                nn.Conv2d(32, 1, 1, bias=False))
        self.conv1x1_ReLU_down4 = nn.Sequential(nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                                nn.Conv2d(32, 1, 1, bias=False))


        self.fuse_predict = nn.Sequential(nn.Conv2d(5, 1, 1, bias=False))

    def forward(self, im):
        H, W = im.size(2), im.size(3)
        lowft, down1, down2, down3, down4 = self.backbone(im)

        # double attention
        down1 = self.attn12(self.attn1(down1))
        down2 = self.attn22(self.attn2(down2))
        down3 = self.attn32(self.attn3(down3))
        down4 = self.attn42(self.attn4(down4))

        lowft = rearrange(lowft, "b (h w) n -> b n h w", h=int(H/4))  # [B, 128, 96, 96]
        down1 = rearrange(down1, "b (h w) n -> b n h w", h=int(H/8))  # [B, 256, 48, 48]
        down2 = rearrange(down2, "b (h w) n -> b n h w", h=int(H/16)) # [B, 512, 24, 24]
        down3 = rearrange(down3, "b (h w) n -> b n h w", h=int(H/32)) # [B, 1024, 12, 12]
        down4 = rearrange(down4, "b (h w) n -> b n h w", h=int(H/32)) # [B, 1024, 12, 12]

        # deep supervision
        attn_lowft, lowft = self.ca_lf(lowft)


        down3_up = F.interpolate(down3, size=down2.size()[2:], mode="bilinear") # [B, 1024, 24, 24]
        down2_up = F.interpolate(down2, size=down1.size()[2:], mode="bilinear") # [B, 512,  48, 48]
        down1_up = F.interpolate(down1, size=lowft.size()[2:], mode="bilinear") # [B, 256,  96, 96]

        p5 = self.conv_p5(down4)
        p4 = self.conv_p5_p4(torch.cat((down4, down3), dim=1))
        p3 = self.conv_p4_p3(torch.cat((down3_up, down2), dim=1))
        p2 = self.conv_p3_p2(torch.cat((down2_up, down1), dim=1))
        p1 = self.conv_p2_p1(torch.cat((down1_up, lowft), dim=1))


        # second stage
        n1 = p1
        p1_down = F.interpolate(p1, size=down1.size()[2:], mode="bilinear") # [B, 256, 48, 48]
        n2 = p2+p1_down
        p2_down = F.interpolate(p2, size=down2.size()[2:], mode="bilinear") # [B, 256, 24, 24]
        n3 = p3+p2_down
        p3_down = F.interpolate(p3, size=down3.size()[2:], mode="bilinear") # [B, 256, 12, 12]
        n4 = p4+p3_down
        p4_down = p4
        n5 = p5+p4_down


        down4_shad = n5  # [B, 256, 12, 12]
        down4_shad0 = down4_shad                                                        # [B, 256, 12, 12]
        down4_shad1 = F.interpolate(down4_shad, size=down2.size()[2:], mode="bilinear") # [B, 256, 24, 24]
        down4_shad2 = F.interpolate(down4_shad, size=down1.size()[2:], mode="bilinear") # [B, 256, 48, 48]
        down4_shad3 = F.interpolate(down4_shad, size=lowft.size()[2:], mode="bilinear") # [B, 256, 96, 96]
        down4_pred = self.conv1x1_ReLU_down4(down4_shad)  # [B, 1, 12, 12]
        shad4 = F.interpolate(down4_pred, size=(H, W), mode="bilinear")  # [B, 1, 384, 384]


        down3_shad = n4  # [B, 256, 12, 12]
        down3_shad0 = F.interpolate(down3_shad, size=down2.size()[2:], mode="bilinear") # [B, 256, 24, 24]
        down3_shad1 = F.interpolate(down3_shad, size=down1.size()[2:], mode="bilinear") # [B, 256, 48, 48]
        down3_shad2 = F.interpolate(down3_shad, size=lowft.size()[2:], mode="bilinear") # [B, 256, 96, 96]
        down3_pred = self.conv1x1_ReLU_down3(torch.cat((down3_shad, down4_shad0), 1))
        shad3 = F.interpolate(down3_pred, size=(H, W), mode="bilinear")  # [B, 1, 384, 384]


        down2_shad = n3  # [B, 512, 24, 24]
        down2_shad0 = F.interpolate(down2_shad, size=down1.size()[2:], mode="bilinear") # [B, 512, 48, 48]
        down2_shad1 = F.interpolate(down2_shad, size=lowft.size()[2:], mode="bilinear") # [B, 512, 96, 96]
        down2_pred = self.conv1x1_ReLU_down2(torch.cat((down2_shad, down3_shad0, down4_shad1), 1))
        shad2 = F.interpolate(down2_pred, size=(H, W), mode="bilinear")  # [B, 1, 384, 384]


        down1_shad = n2  # [B, 256, 48, 48]
        down1_shad0 = F.interpolate(down1_shad, size=lowft.size()[2:], mode="bilinear") # [B, 256, 96, 96]
        down1_pred = self.conv1x1_ReLU_down1(torch.cat((down1_shad, down2_shad0, down3_shad1, down4_shad2), 1))
        shad1 = F.interpolate(down1_pred, size=(H, W), mode="bilinear")  # [B, 1, 384, 384]


        lowft_shad = n1  # [B, 128, 96, 96]
        lowft_pred = self.conv1x1_ReLU_lowft(torch.cat((lowft_shad, down1_shad0, down2_shad1, down3_shad2, down4_shad3), 1))
        shadl = F.interpolate(lowft_pred, size=(H, W), mode="bilinear")  # [B, 1, 384, 384]


        # fuse
        fuse_pred_shad = self.fuse_predict(torch.cat((shad4, shad3, shad2, shad1, shadl), 1))
        return attn_lowft, shad4, shad3, shad2, shad1, shadl, fuse_pred_shad