import pathlib, sys, torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
# Baseline imports torchsummary for its optional historical summary entry.
from models.RHANet_CSSG.rhanet_cssg import RHANetCSSG
from models.RHANet_dep.rhanet_dep import RHANet
from models.RHANet_CSSG_LBAS import CSSGBoundaryAuxTrainerModel

def load_model(kind,checkpoint,device):
 if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; specify --device cpu')
 model={'baseline':RHANet,'cssg':RHANetCSSG,'lbas':CSSGBoundaryAuxTrainerModel}[kind](object()).to(device)
 # Load only checkpoints from a trusted source. No checkpoints are bundled.
 payload=torch.load(checkpoint,map_location=device)
 model.load_state_dict(payload['model_state_dict'],strict=True)
 return model.eval(),payload
