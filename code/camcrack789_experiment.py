"""CamCrack789 fixed protocol for locked baseline/CSSG/LBAS. Formal train is user-invoked only."""
from __future__ import annotations
import argparse,csv,gc,json,logging,random,sys,time,types
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from dataset.CamCrack789_fixed_dataset import CamCrack789FixedDataset
from losses.cross_entropy import WeightedBCELoss
from evaluation_utils import evaluate
from models.RHANet_CSSG_LBAS import CSSGBoundaryAuxTrainerModel,BoundaryAuxiliaryLoss,make_boundary_target
ROOT=Path(__file__).resolve().parents[1];SPL=ROOT/"data"/"CamCrack789"/"splits";LAMBDA_B=.10
SPLITS={"train":SPL/"train_seed2026.txt","val":SPL/"val_seed2026.txt","test":SPL/"test_provided_valid.txt"}
EXPECTED={"baseline":587263,"cssg":587458,"lbas":587651}
def stub_summary():
 try:__import__("torchsummary")
 except ModuleNotFoundError:
  x=types.ModuleType("torchsummary");x.summary=lambda *a,**k:None;sys.modules["torchsummary"]=x
def as_bool(v):
 if isinstance(v,bool):return v
 if v.lower() in {"false","0","no"}:return False
 if v.lower() in {"true","1","yes"}:return True
 raise argparse.ArgumentTypeError("expected true/false")
def arguments():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--mode",choices=("train","val","test"),required=True);p.add_argument("--model",choices=("baseline","cssg","lbas"),required=True)
 p.add_argument("--data-root",default="data/CamCrack789");p.add_argument("--output-dir",required=True);p.add_argument("--checkpoint")
 p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=2026);p.add_argument("--input-size",type=int,default=512)
 p.add_argument("--batch-size",type=int,default=16);p.add_argument("--epochs",type=int,default=100);p.add_argument("--num-workers",type=int,default=4)
 p.add_argument("--validation-interval",type=int,default=10);p.add_argument("--pretrained",type=as_bool,default=False)
 p.add_argument("--threshold-min",type=float,default=.50);p.add_argument("--threshold-max",type=float,default=.99);p.add_argument("--threshold-step",type=float,default=.01)
 p.add_argument("--lambda-b",type=float,default=.10)
 a=p.parse_args()
 if a.seed!=2026 or a.input_size!=512 or a.batch_size!=16 or a.epochs!=100 or a.validation_interval!=10 or a.pretrained:p.error("locked protocol mismatch")
 if (a.threshold_min,a.threshold_max,a.threshold_step)!=(.50,.99,.01):p.error("threshold calibration fixed .50:.01:.99")
 if a.lambda_b!=.10:p.error("lambda_b fixed at .10")
 if a.mode in {"val","test"} and not a.checkpoint:p.error("val/test require checkpoint")
 return a
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def seed_worker(_):
 s=torch.initial_seed()%(2**32);random.seed(s);np.random.seed(s)
def loader(ds,batch,workers,shuffle,seed,drop_last=False):
 return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,drop_last=drop_last,
  worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed),persistent_workers=workers>0)
def dataset(a,split):
 return CamCrack789FixedDataset(a.data_root,Path(a.data_root)/"splits"/SPLITS[split].name,a.input_size,split=="train")
def build_model(kind,device):
 stub_summary()
 if kind=="baseline":
  from models.RHANet_dep.rhanet_dep import RHANet;m=RHANet(object())
 elif kind=="cssg":
  from models.RHANet_CSSG.rhanet_cssg import RHANetCSSG;m=RHANetCSSG(object())
 else:m=CSSGBoundaryAuxTrainerModel(object())
 m=m.to(device);total=sum(p.numel() for p in m.parameters());deploy=sum(p.numel() for p in (m.segmentation_model.parameters() if kind=="lbas" else m.parameters()))
 if total!=EXPECTED[kind]:raise RuntimeError(f"parameter mismatch {kind}: {total} != {EXPECTED[kind]}")
 if kind=="lbas" and deploy!=EXPECTED["cssg"]:raise RuntimeError("LBAS deployment params mismatch")
 return m,total,deploy
def optim(m):
 o=torch.optim.Adam(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False)
 s=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode="min",factor=.95,patience=10,min_lr=1e-6,threshold=.0001,threshold_mode="rel",cooldown=0,eps=1e-8)
 return o,s
def collect(m,dl,crit,dev):
 m.eval();oo=[];tt=[];loss=0.;n=0
 with torch.no_grad():
  for im,t,_ in dl:
   im,t=im.to(dev,non_blocking=True),t.to(dev,non_blocking=True);o=m(im)
   if not torch.isfinite(o).all() or float(o.min())<0 or float(o.max())>1:raise RuntimeError("invalid eval probability")
   if crit is not None:
    z=crit(o,t)
    if not torch.isfinite(z):raise RuntimeError("nonfinite val loss")
    loss+=float(z)
   oo.append(o.cpu());tt.append(t.cpu());n+=1
 return torch.cat(oo),torch.cat(tt),(loss/n if crit is not None else None)
def counts(o,t,th):
 p=o>=th;g=t>=.5
 return tuple(map(int,(torch.logical_and(p,g).sum(),torch.logical_and(p,~g).sum(),torch.logical_and(~p,~g).sum(),torch.logical_and(~p,g).sum())))
def basic(tp,fp,tn,fn):
 q=lambda a,b:float(a/b) if b else 0.;ci=q(tp,tp+fp+fn);bi=q(tn,tn+fp+fn);f=q(2*tp,2*tp+fp+fn)
 return {"accuracy":q(tp+tn,tp+fp+tn+fn),"precision":q(tp,tp+fp),"recall":q(tp,tp+fn),"f1":f,"crack_iou":ci,"background_iou":bi,"miou":(ci+bi)/2,"dice":f,"tp":tp,"fp":fp,"tn":tn,"fn":fn,"prediction_positive_ratio":q(tp+fp,tp+fp+tn+fn)}
def calibrate(o,t):
 best=None;rows=[]
 for i in range(50):
  th=round(.50+i*.01,2);r={"threshold":th,**basic(*counts(o,t,th))};rows.append(r)
  key=(r["f1"],r["crack_iou"],-r["fp"])
  if best is None or key>(best["_key"]):best={**r,"_key":key}
 best.pop("_key");return best,rows
def log_for(out):
 out.mkdir(parents=True,exist_ok=True);x=logging.getLogger("camcrack789");x.handlers.clear();x.setLevel(logging.INFO);fmt=logging.Formatter("%(asctime)s | %(message)s")
 for h in (logging.StreamHandler(),logging.FileHandler(out/"train.log",mode="w",encoding="utf-8")):h.setFormatter(fmt);x.addHandler(h)
 return x
def train(a):
 seed_all(a.seed);dev=torch.device(a.device);out=Path(a.output_dir);log=log_for(out);td,vd=dataset(a,"train"),dataset(a,"val")
 tl=loader(td,a.batch_size,a.num_workers,True,a.seed,True);vl=loader(vd,a.batch_size,a.num_workers,False,a.seed)
 pw=td.compute_pos_weight(3.);segcrit=WeightedBCELoss(pw,reduction="mean",from_logits=False).to(dev);auxcrit=BoundaryAuxiliaryLoss().to(dev) if a.model=="lbas" else None
 m,total,deploy=build_model(a.model,dev);op,sch=optim(m)
 cfg=vars(a).copy();cfg.update({"dataset":"CamCrack789","protocol":"Fixed Reproducible Train/Val/Test Protocol","split_files":{k:str(v) for k,v in SPLITS.items()},"params":total,"deployment_params":deploy,
 "optimizer":"Adam","lr":op.param_groups[0]["lr"],"weight_decay":0.,"scheduler":"ReduceLROnPlateau(factor=.95,patience=10,min_lr=1e-6)","loss":"WeightedBCELoss(from_logits=False)",
 "pos_weight":pw,"normalization":"RGB / 255","image_resize":"bilinear 512x512","mask_binary":"PIL convert('1')","mask_resize":"nearest",
 "augmentation":"Cam repository H/V/180 flips + brightness 1.5/0.5 + contrast 1.5","dataset_specific_augmentation_retained":True,
 "checkpoint_selection":"highest calibrated Val strict crack F1; tie crack IoU then lower FP","test_threshold":"frozen best_val_threshold",
 "lambda_b":.10 if a.model=="lbas" else None,"boundary_target":"3x3 morphological gradient" if a.model=="lbas" else None,
 "boundary_loss":"0.5 BCE + 0.5 Dice" if a.model=="lbas" else None,"boundary_head_deployment":"off" if a.model=="lbas" else None})
 (out/"run_config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
 fields=["epoch","train_seg_loss","train_boundary_loss","train_total_loss","learning_rate","val_loss","val_threshold","val_precision","val_recall","val_f1","val_crack_iou","val_miou","best_f1","best_epoch","best_val_threshold"]
 best=-1.;be=0;bt=None
 with (out/"training_summary.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for ep in range(1,a.epochs+1):
   m.train();ss=bb=tot=0.
   for im,t,_ in tl:
    im,t=im.to(dev,non_blocking=True),t.to(dev,non_blocking=True);op.zero_grad(set_to_none=True)
    if a.model=="lbas":
     seg,boundary,_,_=m(im,return_aux=True);ls=segcrit(seg,t);lb,_=auxcrit(boundary,make_boundary_target(t));z=ls+.10*lb
    else:ls=segcrit(m(im),t);lb=torch.zeros((),device=dev);z=ls
    if not torch.isfinite(z):raise RuntimeError("nonfinite train loss")
    z.backward();op.step();ss+=float(ls);bb+=float(lb);tot+=float(z)
   row={k:"" for k in fields};row.update({"epoch":ep,"train_seg_loss":ss/len(tl),"train_boundary_loss":bb/len(tl) if a.model=="lbas" else "","train_total_loss":tot/len(tl),"learning_rate":op.param_groups[0]["lr"],"best_f1":best,"best_epoch":be,"best_val_threshold":bt})
   if ep%a.validation_interval==0:
    o,t,val_loss=collect(m,vl,segcrit,dev);v,_=calibrate(o,t);sch.step(val_loss)
    if v["f1"]>best:
     best,be,bt=v["f1"],ep,v["threshold"];torch.save({"model_state_dict":m.state_dict(),"model_variant":a.model,"best_epoch":be,"best_val_f1":best,"best_val_threshold":bt},out/"best_model.pth")
    row.update({"val_loss":val_loss,"val_threshold":v["threshold"],"val_precision":v["precision"],"val_recall":v["recall"],"val_f1":v["f1"],"val_crack_iou":v["crack_iou"],"val_miou":v["miou"],"best_f1":best,"best_epoch":be,"best_val_threshold":bt});log.info("epoch %d val %s",ep,json.dumps(v))
   w.writerow(row);f.flush();log.info("epoch %d/%d total %.6f best %.6f@%d T=%s",ep,a.epochs,row["train_total_loss"],best,be,bt)
 (out/"final_metrics.json").write_text(json.dumps({"status":"TRAINING_COMPLETE_AWAITING_EXPLICIT_VAL_TEST","dataset":"CamCrack789","model":a.model,"best_epoch":be,"best_val_threshold":bt,"best_val_f1":best},indent=2),encoding="utf-8")
def run_eval(a):
 seed_all(a.seed);dev=torch.device(a.device);m,total,deploy=build_model(a.model,dev);p=torch.load(a.checkpoint,map_location=dev);m.load_state_dict(p["model_state_dict"],strict=True)
 d=dataset(a,a.mode);dl=loader(d,a.batch_size,a.num_workers,False,a.seed);o,t,_=collect(m,dl,None,dev)
 if a.mode=="val":
  v,_=calibrate(o,t);r={"dataset":"CamCrack789","split":"val","model":a.model,"threshold_source":"Val calibration",**v,**evaluate(o,t,v["threshold"],(2,5,8))}
  if p.get("best_val_threshold") is not None and v["threshold"]!=p["best_val_threshold"]:raise RuntimeError("Val threshold differs from checkpoint")
 else:
  th=p.get("best_val_threshold")
  if th is None:raise RuntimeError("checkpoint lacks frozen best_val_threshold")
  r={"dataset":"CamCrack789","split":"test","model":a.model,"threshold_source":"frozen Val threshold","threshold":th,**evaluate(o,t,th,(2,5,8))}
 r.update({"params":total,"deployment_params":deploy,"boundary_head_called":False if a.model=="lbas" else None,"best_epoch":p.get("best_epoch")})
 print(json.dumps(r,indent=2));path=Path(a.output_dir)/"final_metrics.json";cur=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {};cur[a.mode]=r;path.write_text(json.dumps(cur,indent=2),encoding="utf-8")
def main():
 a=arguments()
 if a.mode=="train":train(a)
 else:run_eval(a)
 gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()
if __name__=="__main__":main()


