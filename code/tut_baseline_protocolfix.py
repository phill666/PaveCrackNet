"""TUT RHACrackNet* baseline protocol-fix training entry. Do not overwrite historical runs."""
from __future__ import annotations
import argparse,csv,json,logging,random,sys,time,types
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from dataset.TUT_dataset import TUTDataset
from losses.cross_entropy import WeightedBCELoss
from evaluation_utils import collect,evaluate

EXPECTED_PARAMS=587263
THRESHOLDS=[x/100 for x in range(50,100)]
def as_bool(v):
 if isinstance(v,bool):return v
 if v.lower() in {"false","0","no"}:return False
 if v.lower() in {"true","1","yes"}:return True
 raise argparse.ArgumentTypeError("expected true/false")
def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=("train",),required=True)
 p.add_argument("--data-root",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--device",default="cuda:0")
 p.add_argument("--seed",type=int,default=2026);p.add_argument("--input-size",type=int,default=512);p.add_argument("--batch-size",type=int,default=16)
 p.add_argument("--epochs",type=int,default=100);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--validation-interval",type=int,default=10)
 p.add_argument("--pretrained",type=as_bool,default=False);p.add_argument("--threshold-min",type=float,default=.50);p.add_argument("--threshold-max",type=float,default=.99);p.add_argument("--threshold-step",type=float,default=.01)
 a=p.parse_args()
 if (a.seed,a.input_size,a.batch_size,a.epochs,a.validation_interval,a.pretrained)!=(2026,512,16,100,10,False):p.error("locked protocol changed")
 if (a.threshold_min,a.threshold_max,a.threshold_step)!=(.50,.99,.01):p.error("threshold protocol fixed at .50:.01:.99")
 out=Path(a.output_dir)
 if out.exists() and any(out.iterdir()):p.error("output directory exists and is not empty; refusing overwrite")
 return a
def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def seed_worker(_):
 s=torch.initial_seed()%2**32;random.seed(s);np.random.seed(s)
def loader(ds,b,w,shuffle,seed,drop=False):
 return DataLoader(ds,batch_size=b,shuffle=shuffle,num_workers=w,pin_memory=True,drop_last=drop,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed),persistent_workers=w>0)
def model_for(dev):
 try:__import__("torchsummary")
 except ModuleNotFoundError:
  q=types.ModuleType("torchsummary");q.summary=lambda *x,**y:None;sys.modules["torchsummary"]=q
 from models.RHANet_dep.rhanet_dep import RHANet
 m=RHANet(object()).to(dev);n=sum(p.numel() for p in m.parameters())
 if n!=EXPECTED_PARAMS:raise RuntimeError(f"parameter mismatch {n} != {EXPECTED_PARAMS}")
 return m
def counts(o,t,th):
 p=o>=th;g=t>=.5
 return tuple(map(int,(torch.logical_and(p,g).sum(),torch.logical_and(p,~g).sum(),torch.logical_and(~p,~g).sum(),torch.logical_and(~p,g).sum())))
def metrics(tp,fp,tn,fn):
 q=lambda a,b:float(a/b) if b else 0.;ci=q(tp,tp+fp+fn);bi=q(tn,tn+fp+fn);f=q(2*tp,2*tp+fp+fn)
 return {"accuracy":q(tp+tn,tp+fp+tn+fn),"precision":q(tp,tp+fp),"recall":q(tp,tp+fn),"f1":f,"crack_iou":ci,"background_iou":bi,"miou":(ci+bi)/2,"dice":f,"tp":tp,"fp":fp,"tn":tn,"fn":fn,"prediction_positive_ratio":q(tp+fp,tp+fp+tn+fn)}
def better(a,b):
 return b is None or (a["f1"],a["crack_iou"],-a["fp"])>(b["f1"],b["crack_iou"],-b["fp"])
def calibrate(o,t):
 rows=[];best=None
 for th in THRESHOLDS:
  r={"threshold":th,**metrics(*counts(o,t,th))};rows.append(r)
  if better(r,best):best=r
 return best,rows
def optim(m):
 o=torch.optim.Adam(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False)
 s=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode="min",factor=.95,patience=10,min_lr=1e-6,threshold=.0001,threshold_mode="rel",cooldown=0,eps=1e-8)
 return o,s
def log_for(out):
 x=logging.getLogger("tut_baseline_protocolfix");x.handlers.clear();x.setLevel(logging.INFO);fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
 for h in (logging.StreamHandler(),logging.FileHandler(out/"train.log",mode="w",encoding="utf-8")):h.setFormatter(fmt);x.addHandler(h)
 return x
def main():
 a=args();seed_all(a.seed);dev=torch.device(a.device);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);log=log_for(out)
 tr=TUTDataset(a.data_root,"train",a.input_size,True);va=TUTDataset(a.data_root,"val",a.input_size,False);te=TUTDataset(a.data_root,"test",a.input_size,False)
 tl=loader(tr,a.batch_size,a.num_workers,True,a.seed,True);vl=loader(va,a.batch_size,a.num_workers,False,a.seed);tel=loader(te,a.batch_size,a.num_workers,False,a.seed)
 crit=WeightedBCELoss(tr.compute_pos_weight(3.0),reduction="mean",from_logits=False).to(dev);m=model_for(dev);op,sch=optim(m)
 cfg=vars(a).copy();cfg.update({"dataset":"TUT","model":"RHACrackNet*","class":"RHANet","source":"code/models/RHANet_dep/rhanet_dep.py","params":EXPECTED_PARAMS,"trainable_params":EXPECTED_PARAMS,"optimizer":"Adam","lr":op.param_groups[0]["lr"],"betas":[.9,.999],"eps":1e-8,"weight_decay":0.,"amsgrad":False,"scheduler":"ReduceLROnPlateau(mode=min,factor=.95,patience=10,min_lr=1e-6)","warmup":"none","augmentation":"TUTDataset official H/V/180 flips + brightness 1.5/0.5 + contrast 1.5","segmentation_loss":"WeightedBCELoss(from_logits=False)","pos_weight":tr.compute_pos_weight(3.0),"checkpoint_selection":"highest calibrated Val strict crack F1; tie crack IoU; tie lower FP","test_threshold":"frozen best_val_threshold","periodic_checkpoints":False})
 (out/"run_config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
 fields=["epoch","train_loss","learning_rate","val_loss","val_threshold","val_precision","val_recall","val_f1","val_crack_iou","val_miou","val_fp","val_fn","best_f1","best_epoch","best_val_threshold"];valrows=[];best=None;be=0;bt=None;started=time.perf_counter()
 with (out/"training_summary.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for ep in range(1,a.epochs+1):
   m.train();loss_sum=0.
   for im,t in tl:
    im,t=im.to(dev,non_blocking=True),t.to(dev,non_blocking=True);op.zero_grad(set_to_none=True);z=crit(m(im),t)
    if not torch.isfinite(z):raise RuntimeError("non-finite loss")
    z.backward();op.step();loss_sum+=float(z)
   row={k:"" for k in fields};row.update({"epoch":ep,"train_loss":loss_sum/len(tl),"learning_rate":op.param_groups[0]["lr"],"best_f1":best["f1"] if best else -1.,"best_epoch":be,"best_val_threshold":bt})
   if ep%a.validation_interval==0:
    prob,target,vloss=collect(m,vl,crit,dev);v,_=calibrate(prob,target);sch.step(vloss);vrow={"epoch":ep,**v};valrows.append(vrow)
    if better(v,best):
     best,be,bt=v,ep,v["threshold"];torch.save({"model_state_dict":m.state_dict(),"model_variant":"baseline","best_epoch":be,"best_val_f1":best["f1"],"best_val_threshold":bt},out/"best_model.pth")
    row.update({"val_loss":vloss,"val_threshold":v["threshold"],"val_precision":v["precision"],"val_recall":v["recall"],"val_f1":v["f1"],"val_crack_iou":v["crack_iou"],"val_miou":v["miou"],"val_fp":v["fp"],"val_fn":v["fn"],"best_f1":best["f1"],"best_epoch":be,"best_val_threshold":bt});log.info("epoch %d calibrated val %s",ep,json.dumps(v))
   w.writerow(row);f.flush();log.info("epoch %d/%d loss %.6f best %.6f@%d T=%s",ep,a.epochs,row["train_loss"],best["f1"] if best else -1.,be,bt)
 with (out/"validation_epoch_metrics.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(valrows[0]));w.writeheader();w.writerows(valrows)
 payload=torch.load(out/"best_model.pth",map_location=dev);m.load_state_dict(payload["model_state_dict"],strict=True);m.eval();vp,vt,_=collect(m,vl,None,dev);reval,_=calibrate(vp,vt)
 if reval["threshold"]!=bt:raise RuntimeError("frozen Val threshold verification failed")
 tp,tt,_=collect(m,tel,None,dev);test={"dataset":"TUT","split":"test","threshold_source":"frozen Val threshold","threshold":bt,**evaluate(tp,tt,bt,(2,5,8)),"params":EXPECTED_PARAMS,"best_epoch":be}
 final={"dataset":"TUT","seed":a.seed,"input_size":a.input_size,"batch_size":a.batch_size,"epochs":a.epochs,"best_epoch":be,"best_val_threshold":bt,"best_val_f1":best["f1"],"val":{**best,**evaluate(vp,vt,bt,(2,5,8))},"test":test,"total_training_time_sec":time.perf_counter()-started}
 (out/"final_metrics.json").write_text(json.dumps(final,indent=2),encoding="utf-8");print(json.dumps(final,indent=2))
if __name__=="__main__":main()

