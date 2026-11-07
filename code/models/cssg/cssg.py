"""CSSG V1: Cross Scale Semantic Suppression Gate."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CSSG(nn.Module):
    """Suppress shallow responses that lack deep semantic support."""

    def __init__(self, shallow_channels, deep_channels, beta_init=0.1):
        super().__init__()
        if not 0.0 < beta_init < 1.0:
            raise ValueError("beta_init must be strictly between zero and one")
        self.shallow_score = nn.Conv2d(shallow_channels, 1, kernel_size=1, bias=True)
        self.deep_score = nn.Conv2d(deep_channels, 1, kernel_size=1, bias=True)
        self.beta_logit = nn.Parameter(torch.tensor(math.log(beta_init / (1.0 - beta_init))))
        self.collect_stats_once = False
        self.last_stats = None

    @property
    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def request_stats_once(self):
        self.collect_stats_once = True
        self.last_stats = None

    def forward(self, shallow, deep):
        if shallow.ndim != 4 or deep.ndim != 4:
            raise ValueError("CSSG expects two NCHW feature tensors")
        deep_up = F.interpolate(deep, size=shallow.shape[-2:], mode="bilinear", align_corners=None)
        p_s = torch.sigmoid(self.shallow_score(shallow))
        p_d = torch.sigmoid(self.deep_score(deep_up))
        m_fp = p_s * (1.0 - p_d)
        output = shallow * (1.0 - self.beta * m_fp)

        if self.collect_stats_once:
            def summary(tensor):
                detached = tensor.detach().float()
                return {"min": detached.min().item(), "max": detached.max().item(),
                        "mean": detached.mean().item(), "std": detached.std(unbiased=False).item()}
            denominator = shallow.detach().abs().mean().item() + 1e-8
            relative_change = (output.detach() - shallow.detach()).abs().mean().item() / denominator
            self.last_stats = {"p_s": summary(p_s), "p_d": summary(p_d),
                               "m_fp": summary(m_fp), "beta": self.beta.detach().item(),
                               "relative_change": relative_change}
            self.collect_stats_once = False
        return output
