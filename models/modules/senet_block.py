import torch.nn as nn
import MinkowskiEngine as ME

from mix3d.models.modules.common import ConvType, NormType
from mix3d.models.modules.resnet_block import BasicBlock, Bottleneck


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, D=-1):
        # Global coords does not require coords_key
        super().__init__()
        self.fc = nn.Sequential(
            ME.MinkowskiLinear(channel, channel // reduction),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(channel // reduction, channel),
            ME.MinkowskiSigmoid(),
        )
        self.pooling = ME.MinkowskiGlobalPooling(dimension=D)
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication(dimension=D)

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        return self.broadcast_mul(x, y)


class SEBasicBlock(BasicBlock):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        conv_type=ConvType.HYPERCUBE,
        reduction=16,
        D=-1,
    ):
        super().__init__(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            conv_type=conv_type,
            D=D,
        )
        self.se = SELayer(planes, reduction=reduction, D=D)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBasicBlockSN(SEBasicBlock):
    NORM_TYPE = NormType.SPARSE_SWITCH_NORM


class SEBasicBlockIN(SEBasicBlock):
    NORM_TYPE = NormType.SPARSE_INSTANCE_NORM


class SEBasicBlockLN(SEBasicBlock):
    NORM_TYPE = NormType.SPARSE_LAYER_NORM


class SEBottleneck(Bottleneck):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        conv_type=ConvType.HYPERCUBE,
        D=3,
        reduction=16,
    ):
        super().__init__(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            conv_type=conv_type,
            D=D,
        )
        self.se = SELayer(planes * self.expansion, reduction=reduction, D=D)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneckSN(SEBottleneck):
    NORM_TYPE = NormType.SPARSE_SWITCH_NORM


class SEBottleneckIN(SEBottleneck):
    NORM_TYPE = NormType.SPARSE_INSTANCE_NORM


class SEBottleneckLN(SEBottleneck):
    NORM_TYPE = NormType.SPARSE_LAYER_NORM
