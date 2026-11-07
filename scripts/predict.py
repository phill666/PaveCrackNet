import argparse, pathlib
import numpy as np
from PIL import Image
import torch
from runtime import load_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--model',choices=['baseline','cssg','lbas'],default='lbas');p.add_argument('--device',default='cpu');p.add_argument('--threshold',type=float);a=p.parse_args()
 m,payload=load_model(a.model,a.checkpoint,a.device);th=a.threshold if a.threshold is not None else payload.get('best_val_threshold')
 if th is None or not 0<=th<=1:raise ValueError('Supply a valid frozen validation threshold')
 source=pathlib.Path(a.input);files=sorted(q for q in source.iterdir() if q.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif')) if source.is_dir() else [source]
 out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
 for i,q in enumerate(files):
  with Image.open(q) as im:rgb=im.convert('RGB').resize((512,512),Image.Resampling.BILINEAR)
  arr=np.asarray(rgb,dtype=np.float32)/255.;x=torch.from_numpy(arr.transpose(2,0,1).copy())[None].to(a.device)
  with torch.no_grad():prob=m(x)[0,0].cpu().numpy()
  mask=(prob>=th).astype(np.uint8)*255;name=f'{i:06d}_{q.stem}'
  Image.fromarray(mask).save(out/(name+'_mask.png'))
  overlay=np.asarray(rgb).copy();overlay[mask>0]=(overlay[mask>0]*.5+np.array([255,0,0])*.5).astype(np.uint8)
  Image.fromarray(overlay).save(out/(name+'_overlay.png'))
if __name__=='__main__':main()
