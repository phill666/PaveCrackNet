import argparse, pathlib, json
import torch
from runtime import load_model
from dataset.TUT_dataset import TUTDataset
from dataset.CamCrack789_fixed_dataset import CamCrack789FixedDataset
from dataset.PYCrackDB_fixed_dataset import PYCrackDBFixedDataset
from evaluation_utils import evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--calibrate-val',action='store_true');p.add_argument('--dataset',choices=['tut','camcrack789','pycrackdb'],required=True);p.add_argument('--data-root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--model',choices=['baseline','cssg','lbas'],default='lbas');p.add_argument('--device',default='cpu');p.add_argument('--threshold',type=float);p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--batch-size',type=int,default=16);a=p.parse_args()
 m,payload=load_model(a.model,a.checkpoint,a.device);th=a.threshold if a.threshold is not None else payload.get('best_val_threshold')
 if a.calibrate_val and a.split!='val':raise ValueError('Calibration is validation-only')
 if a.calibrate_val:th=.5
 if th is None or not 0<=th<=1:raise ValueError('Supply a frozen validation threshold; no Test search is performed')
 if a.dataset=='tut':ds=TUTDataset(a.data_root,a.split,512,False)
 else:
  filename='test_provided_valid.txt' if a.dataset=='camcrack789' and a.split=='test' else a.split+'_seed2026.txt'
  cls=CamCrack789FixedDataset if a.dataset=='camcrack789' else PYCrackDBFixedDataset
  ds=cls(a.data_root,pathlib.Path(a.data_root)/'splits'/filename,512,False)
 oo=[];tt=[]
 with torch.no_grad():
  for batch in torch.utils.data.DataLoader(ds,batch_size=a.batch_size,shuffle=False):oo.append(m(batch[0].to(a.device)).cpu());tt.append(batch[1])
 o,t=torch.cat(oo),torch.cat(tt)
 if a.calibrate_val:
  candidates=[(h/100,evaluate(o,t,h/100)) for h in range(5,100)]
  th=max(candidates,key=lambda z:(z[1]['f1'],z[1]['crack_iou'],-z[1]['fp']))[0]
 print(json.dumps({'threshold':th,**evaluate(o,t,th,(2,5,8))},indent=2))
if __name__=='__main__':main()
