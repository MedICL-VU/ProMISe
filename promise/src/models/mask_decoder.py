import torch
import torch.nn as nn
import torch.nn.functional as F
class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())

        self.up2 = nn.ConvTranspose3d(mlahead_channels, mlahead_channels, 2, stride=2)
        self.up3 = nn.ConvTranspose3d(mlahead_channels, mlahead_channels, 2, stride=2)
        self.up4 = nn.ConvTranspose3d(mlahead_channels, mlahead_channels, 2, stride=2)
        self.up5 = nn.ConvTranspose3d(mlahead_channels, mlahead_channels, 2, stride=2)


    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5, scale_factor):
        # head2 = self.head2(mla_p2)
        head2 = self.up2(self.head2(mla_p2))
        head3 = self.up3(self.head3(mla_p3))
        head4 = self.up4(self.head4(mla_p4))
        head5 = self.up5(self.head5(mla_p5))
        return torch.cat([head2, head3, head4, head5], dim=1)


class VIT_MLAHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, num_classes=2,
                 norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLAHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels

        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)


        self.head = nn.Sequential(nn.Conv3d(5 * mlahead_channels + 1, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, num_classes, 3, padding=1, bias=False))

        self.image_feature_extractor = nn.Sequential(
                     nn.Conv3d(1, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     )


    def forward(self, inputs, scale_factor1, scale_factor2):
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3], scale_factor=scale_factor1)
        image_features = self.image_feature_extractor(inputs[-1])
        x = torch.cat([x, image_features, inputs[-1]], dim=1)
        x = self.head(x)
        x = F.interpolate(x, scale_factor=scale_factor2, mode='trilinear', align_corners=True)
        return x