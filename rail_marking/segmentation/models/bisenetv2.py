#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["BiSeNetV2"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        need_activation=True,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self._need_activation = need_activation

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        if self._need_activation:
            feat = self.relu(feat)
        return feat


class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBlock(in_chan=3, out_chan=64, kernel_size=3, stride=2),
            ConvBlock(in_chan=64, out_chan=64, kernel_size=3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBlock(in_chan=64, out_chan=64, kernel_size=3, stride=2),
            ConvBlock(in_chan=64, out_chan=64, kernel_size=3, stride=1),
            ConvBlock(in_chan=64, out_chan=64, kernel_size=3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBlock(in_chan=64, out_chan=128, kernel_size=3, stride=2),
            ConvBlock(in_chan=128, out_chan=128, kernel_size=3, stride=1),
            ConvBlock(in_chan=128, out_chan=128, kernel_size=3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)

        return feat


class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBlock(in_chan=3, out_chan=16, kernel_size=3, stride=2)
        self.left = nn.Sequential(
            ConvBlock(in_chan=16, out_chan=8, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_chan=8, out_chan=16, kernel_size=3, stride=2),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBlock(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class ContextEmbeddingBlock(nn.Module):
    def __init__(self):
        super(ContextEmbeddingBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBlock(in_chan=128, out_chan=128, kernel_size=1, stride=1, padding=0)
        self.conv_last = ConvBlock(in_chan=128, out_chan=128, kernel_size=3, stride=1, need_activation=False)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GatherExpandLayerS1(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GatherExpandLayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBlock(in_chan=in_chan, out_chan=in_chan, kernel_size=3, stride=1)
        self.dwconv = ConvBlock(
            in_chan=in_chan,
            out_chan=mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_chan,
            bias=False,
            need_activation=False,
        )
        self.conv2 = ConvBlock(
            mid_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            need_activation=False,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GatherExpandLayerS2(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GatherExpandLayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBlock(in_chan=in_chan, out_chan=in_chan, kernel_size=3, stride=1)
        self.dwconv1 = ConvBlock(
            in_chan=in_chan,
            out_chan=mid_chan,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_chan,
            bias=False,
            need_activation=False,
        )
        self.dwconv2 = ConvBlock(
            in_chan=mid_chan,
            out_chan=mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=mid_chan,
            bias=False,
            need_activation=False,
        )
        self.conv2 = ConvBlock(
            in_chan=mid_chan,
            out_chan=out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            need_activation=False,
        )
        self.shortcut = nn.Sequential(
            ConvBlock(
                in_chan=in_chan,
                out_chan=in_chan,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_chan,
                bias=False,
                need_activation=False,
            ),
            ConvBlock(
                in_chan=in_chan,
                out_chan=out_chan,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                need_activation=False,
            ),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SemanticBranch(nn.Module):
    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GatherExpandLayerS2(16, 32),
            GatherExpandLayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GatherExpandLayerS2(32, 64),
            GatherExpandLayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GatherExpandLayerS2(64, 128),
            GatherExpandLayerS1(128, 128),
            GatherExpandLayerS1(128, 128),
            GatherExpandLayerS1(128, 128),
        )
        self.S5_5 = ContextEmbeddingBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BilateralGuidedAggregationLayer(nn.Module):
    def __init__(self):
        super(BilateralGuidedAggregationLayer, self).__init__()

        # Detail Branch
        self.left1 = nn.Sequential(
            ConvBlock(
                in_chan=128,
                out_chan=128,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=128,
                bias=False,
                need_activation=False,
            ),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            ConvBlock(
                128,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                need_activation=False,
            ),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )

        # Semantic Branch
        self.right1 = ConvBlock(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            need_activation=False,
        )
        self.right2 = nn.Sequential(
            ConvBlock(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=128,
                bias=False,
                need_activation=False,
            ),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.conv = ConvBlock(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            need_activation=False,
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = F.interpolate(right1, size=dsize, mode="bilinear", align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(right, size=dsize, mode="bilinear", align_corners=True)
        out = self.conv(left + right)
        return out


class SegmentHead(nn.Module):
    def __init__(self, in_chan, seghead_ratio, n_classes, dropout_rate=0.1):
        super(SegmentHead, self).__init__()
        mid_chan = in_chan * seghead_ratio
        self.conv = ConvBlock(in_chan, mid_chan, 3, stride=1)
        self.drop_out = nn.Dropout(dropout_rate)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, size=None):
        feat = self.conv(x)
        feat = self.drop_out(feat)
        feat = self.conv_out(feat)
        if size:
            feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=True)

        return feat


class BiSeNetV2(nn.Module):
    def __init__(self, n_classes, seghead_ratio=6):
        super(BiSeNetV2, self).__init__()
        self.detail = DetailBranch()
        self.segment = SemanticBranch()
        self.bga = BilateralGuidedAggregationLayer()

        self.head = SegmentHead(128, seghead_ratio, n_classes)
        if self.training:
            self.aux2 = SegmentHead(16, seghead_ratio, n_classes)
            self.aux3 = SegmentHead(32, seghead_ratio, n_classes)
            self.aux4 = SegmentHead(64, seghead_ratio, n_classes)
            self.aux5_4 = SegmentHead(128, seghead_ratio, n_classes)

            self.init_weights()

    def forward(self, x):
        size = x.size()[2:]
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)

        feat_head = self.bga(feat_d, feat_s)

        logits = self.head(feat_head, size)
        if self.training:
            logits_aux2 = self.aux2(feat2, size)
            logits_aux3 = self.aux3(feat3, size)
            logits_aux4 = self.aux4(feat4, size)
            logits_aux5_4 = self.aux5_4(feat5_4, size)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        else:
            return logits

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, "last_bn") and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
