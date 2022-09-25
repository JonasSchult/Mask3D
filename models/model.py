from MinkowskiEngine import MinkowskiNetwork


class Model(MinkowskiNetwork):
    """
  Base network for all sparse convnet

  By default, all networks are segmentation networks.
  """

    OUT_PIXEL_DIST = -1

    def __init__(self, in_channels, out_channels, config, D, **kwargs):
        super().__init__(D)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = config


class HighDimensionalModel(Model):
    """
  Base network for all spatio (temporal) chromatic sparse convnet
  """

    def __init__(self, in_channels, out_channels, config, D, **kwargs):
        assert D > 4, "Num dimension smaller than 5"
        super().__init__(in_channels, out_channels, config, D, **kwargs)
