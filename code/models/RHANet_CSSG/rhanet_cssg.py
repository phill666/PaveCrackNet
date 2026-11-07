"""RHACrackNet* with exactly one CSSG V1 on the 1/4 encoder skip."""
import sys
import types

try:
    __import__("torchsummary")
except ModuleNotFoundError:
    stub = types.ModuleType("torchsummary")
    stub.summary = lambda *args, **kwargs: None
    sys.modules["torchsummary"] = stub

from models.RHANet_dep.rhanet_dep import RHANet
from models.cssg import CSSG


class RHANetCSSG(RHANet):
    """Light RHACrackNet with one cross-scale semantic suppression gate."""

    def __init__(self, config):
        super().__init__(config)
        self.cssg = CSSG(shallow_channels=64, deep_channels=128, beta_init=0.1)

    def forward(self, input_):
        input_ = self._same_padding(input_)
        init_conv = self.init_conv(input_)
        encode_1 = self.encode_1(init_conv)
        encode_2 = self.encode_2(encode_1)
        encode_3 = self.encode_3(encode_2)
        encode_4 = self.encode_4(encode_3)
        encode_4_mid = encode_4
        encode_4 = self.resencode_4(encode_4)
        encode_4 = self.resencode_4_1(encode_4)
        encode_4_end = self.attention_GAP(encode_4_mid, encode_4)

        encode_2 = self.cssg(encode_2, encode_4_end)
        decode_1 = self.decode_1(encode_4_end, encode_3)
        decode_2 = self.decode_2(decode_1, encode_2)
        decode_3 = self.decode_3(decode_2, encode_1)
        decode_4 = self.decode_4(decode_3, init_conv)
        output = self.final_conv(decode_4)
        return output[:, :, 0:self.size_x, 0:self.size_y]
