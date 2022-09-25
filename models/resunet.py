import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine import MinkowskiReLU

from models.resnet import ResNetBase, get_norm
from models.modules.common import ConvType, NormType, conv, conv_tr
from models.modules.resnet_block import BasicBlock, Bottleneck, BasicBlockINBN


class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = (64, 128, 256, 512, 256, 128, 128)
    DILATIONS = (1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2)
    INIT_DIM = 64
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super().__init__(in_channels, out_channels, config, D)

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv1p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )

        self.bn1 = get_norm(self.NORM_TYPE, self.PLANES[0], D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv2p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv3p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv4p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr4p8s2 = conv_tr(
            self.inplanes,
            self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr4 = get_norm(
            self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr5p4s2 = conv_tr(
            self.inplanes,
            self.PLANES[5],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr5 = get_norm(
            self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.convtr6p2s2 = conv_tr(
            self.inplanes,
            self.PLANES[6],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr6 = get_norm(
            self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum
        )
        self.relu = MinkowskiReLU(inplace=True)

        self.final = nn.Sequential(
            conv(
                self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion,
                512,
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=False,
                D=D,
            ),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(),
            conv(
                512, out_channels, kernel_size=1, stride=1, dilation=1, bias=True, D=D
            ),
        )

    def forward(self, x):
        out = self.conv1p1s1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out_b1p1 = self.block1(out)

        out = self.conv2p1s2(out_b1p1)
        out = self.bn2(out)
        out = self.relu(out)

        out_b2p2 = self.block2(out)

        out = self.conv3p2s2(out_b2p2)
        out = self.bn3(out)
        out = self.relu(out)

        out_b3p4 = self.block3(out)

        out = self.conv4p4s2(out_b3p4)
        out = self.bn4(out)
        out = self.relu(out)

        # pixel_dist=8
        out = self.block4(out)

        out = self.convtr4p8s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = me.cat(out, out_b3p4)
        out = self.block5(out)

        out = self.convtr5p4s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = me.cat(out, out_b2p2)
        out = self.block6(out)

        out = self.convtr6p2s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = me.cat(out, out_b1p1)
        return self.final(out)


class ResUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1)


class ResUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2)


class ResUNet18INBN(ResUNet18):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM
    BLOCK = BasicBlockINBN


class ResUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3, 2, 2)


class ResUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3, 2, 2)


class ResUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3, 2, 2)


class ResUNet14D(ResUNet14):
    PLANES = (64, 128, 256, 512, 512, 512, 512)


class ResUNet18D(ResUNet18):
    PLANES = (64, 128, 256, 512, 512, 512, 512)


class ResUNet34D(ResUNet34):
    PLANES = (64, 128, 256, 512, 512, 512, 512)


class ResUNet34E(ResUNet34):
    INIT_DIM = 32
    PLANES = (32, 64, 128, 256, 128, 64, 64)


class ResUNet34F(ResUNet34):
    INIT_DIM = 32
    PLANES = (32, 64, 128, 256, 128, 64, 32)


class MinkUNetHyper(MinkUNetBase):
    BLOCK = None
    PLANES = (64, 128, 256, 512, 256, 128, 128)
    DILATIONS = (1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2)
    INIT_DIM = 64
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(MinkUNetBase, self).__init__(in_channels, out_channels, config, D)

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv1p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )

        self.bn1 = get_norm(self.NORM_TYPE, self.PLANES[0], D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv2p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv3p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv4p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.pool_tr4 = ME.MinkowskiPoolingTranspose(
            kernel_size=8, stride=8, dimension=D
        )
        _ = self.inplanes
        self.convtr4p8s2 = conv_tr(
            self.inplanes,
            self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr4 = get_norm(
            self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.pool_tr5 = ME.MinkowskiPoolingTranspose(
            kernel_size=4, stride=4, dimension=D
        )
        out_pool5 = self.inplanes
        self.convtr5p4s2 = conv_tr(
            self.inplanes,
            self.PLANES[5],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr5 = get_norm(
            self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum
        )

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )
        self.pool_tr6 = ME.MinkowskiPoolingTranspose(
            kernel_size=2, stride=2, dimension=D
        )
        out_pool6 = self.inplanes
        self.convtr6p2s2 = conv_tr(
            self.inplanes,
            self.PLANES[6],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D,
        )
        self.bntr6 = get_norm(
            self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum
        )

        self.relu = MinkowskiReLU(inplace=True)

        self.final = nn.Sequential(
            conv(
                out_pool5
                + out_pool6
                + self.PLANES[6]
                + self.PLANES[0] * self.BLOCK.expansion,
                512,
                kernel_size=1,
                bias=False,
                D=D,
            ),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(),
            conv(512, out_channels, kernel_size=1, bias=True, D=D),
        )

    def forward(self, x):
        out = self.conv1p1s1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out_b1p1 = self.block1(out)

        out = self.conv2p1s2(out_b1p1)
        out = self.bn2(out)
        out = self.relu(out)

        out_b2p2 = self.block2(out)

        out = self.conv3p2s2(out_b2p2)
        out = self.bn3(out)
        out = self.relu(out)

        out_b3p4 = self.block3(out)

        out = self.conv4p4s2(out_b3p4)
        out = self.bn4(out)
        out = self.relu(out)

        # pixel_dist=8
        out = self.block4(out)

        out = self.convtr4p8s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = me.cat(out, out_b3p4)
        out = self.block5(out)
        out_5 = self.pool_tr5(out)

        out = self.convtr5p4s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = me.cat(out, out_b2p2)
        out = self.block6(out)
        out_6 = self.pool_tr6(out)

        out = self.convtr6p2s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = me.cat(out, out_b1p1, out_6, out_5)
        return self.final(out)


class MinkUNetHyper14INBN(MinkUNetHyper):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM
    BLOCK = BasicBlockINBN


class STMinkUNetBase(MinkUNetBase):

    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
        super().__init__(in_channels, out_channels, config, D, **kwargs)


class STResUNet14(STMinkUNetBase, ResUNet14):
    pass


class STResUNet18(STMinkUNetBase, ResUNet18):
    pass


class STResUNet34(STMinkUNetBase, ResUNet34):
    pass


class STResUNet50(STMinkUNetBase, ResUNet50):
    pass


class STResUNet101(STMinkUNetBase, ResUNet101):
    pass


class STResTesseractUNetBase(STMinkUNetBase):
    CONV_TYPE = ConvType.HYPERCUBE


class STResTesseractUNet14(STResTesseractUNetBase, ResUNet14):
    pass


class STResTesseractUNet18(STResTesseractUNetBase, ResUNet18):
    pass


class STResTesseractUNet34(STResTesseractUNetBase, ResUNet34):
    pass


class STResTesseractUNet50(STResTesseractUNetBase, ResUNet50):
    pass


class STResTesseractUNet101(STResTesseractUNetBase, ResUNet101):
    pass
