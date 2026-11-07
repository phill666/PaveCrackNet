"""Training-only lightweight boundary auxiliary supervision for locked CSSG."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from models.RHANet_CSSG.rhanet_cssg import RHANetCSSG

def make_boundary_target(mask: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    binary=(mask>0.5).to(mask.dtype)
    dilated=F.max_pool2d(binary,kernel_size=3,stride=1,padding=1)
    eroded=1.0-F.max_pool2d(1.0-binary,kernel_size=3,stride=1,padding=1)
    boundary=(dilated-eroded).clamp_(0.0,1.0)
    if valid_mask is not None:
        boundary=boundary*(valid_mask>0).to(boundary.dtype)
    return boundary

class LightweightBoundaryHead(nn.Module):
    def __init__(self, channels: int=16):
        super().__init__()
        self.dw=nn.Conv2d(channels,channels,3,stride=1,padding=1,groups=channels,bias=False)
        self.bn=nn.BatchNorm2d(channels)
        self.act=nn.ReLU(inplace=True)
        self.pw=nn.Conv2d(channels,1,1,bias=True)
    def forward(self, feature: torch.Tensor, output_size=None) -> tuple[torch.Tensor,torch.Tensor]:
        raw=self.pw(self.act(self.bn(self.dw(feature))))
        up=raw if output_size is None or raw.shape[-2:]==tuple(output_size) else F.interpolate(
            raw,size=output_size,mode="bilinear",align_corners=None)
        return raw,up

class BoundaryAuxiliaryLoss(nn.Module):
    def __init__(self, eps: float=1e-6):
        super().__init__();self.eps=float(eps)
    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        positive=target.sum()
        negative=target.numel()-positive
        pos_weight=(negative/positive.clamp_min(1.0)).clamp(1.0,20.0).detach()
        bce=F.binary_cross_entropy_with_logits(logits,target,pos_weight=pos_weight)
        probability=torch.sigmoid(logits)
        intersection=(probability*target).sum()
        dice=(2.0*intersection+self.eps)/(probability.sum()+target.sum()+self.eps)
        dice_loss=1.0-dice
        boundary=0.5*bce+0.5*dice_loss
        return boundary,{"bce":bce,"dice":dice_loss,"pos_weight":pos_weight}

class CSSGBoundaryAuxTrainerModel(nn.Module):
    """The auxiliary head is called only when return_aux=True."""
    def __init__(self, config=None):
        super().__init__()
        self.segmentation_model=RHANetCSSG(config if config is not None else object())
        self.boundary_head=LightweightBoundaryHead(16)
        self._decode3_feature=None
        self._hook=self.segmentation_model.decode_3.register_forward_hook(self._capture)
    def _capture(self,_module,_inputs,output):
        self._decode3_feature=output
    def forward(self,x: torch.Tensor,return_aux: bool=False):
        self._decode3_feature=None
        segmentation=self.segmentation_model(x)
        if not return_aux:
            self._decode3_feature=None
            return segmentation
        feature=self._decode3_feature
        if feature is None:
            raise RuntimeError("decode_3 feature hook did not fire")
        raw,up=self.boundary_head(feature,segmentation.shape[-2:])
        self._decode3_feature=None
        return segmentation,up,raw,feature
