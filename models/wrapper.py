import random

from torch.nn import Module
from MinkowskiEngine import SparseTensor


class Wrapper(Module):
    """
  Wrapper for the segmentation networks.
  """

    OUT_PIXEL_DIST = -1

    def __init__(self, NetClass, in_nchannel, out_nchannel, config):
        super().__init__()
        self.initialize_filter(NetClass, in_nchannel, out_nchannel, config)

    def initialize_filter(self, NetClass, in_nchannel, out_nchannel, config):
        raise NotImplementedError("Must initialize a model and a filter")

    def forward(self, x, coords, colors=None):
        soutput = self.model(x)

        # During training, make the network invariant to the filter
        if not self.training or random.random() < 0.5:
            # Filter requires the model to finish the forward pass
            wrapper_coords = self.filter.initialize_coords(self.model, coords, colors)
            finput = SparseTensor(soutput.F, wrapper_coords)
            soutput = self.filter(finput)
        return soutput
