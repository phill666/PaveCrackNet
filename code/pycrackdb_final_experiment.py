"""PY-CrackDB fixed final verification for locked baseline/CSSG/LBAS."""
from __future__ import annotations
import argparse,csv,gc,json,logging,random,sys,time,types
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from dataset.PYCrackDB_fixed_dataset import PYCrackDBFixedDataset
from losses.cross_entropy import WeightedBCELoss
from evaluation_utils import evaluate
from models.RHANet_CSSG_LBAS import CSSGBoundaryAuxTrainerModel,BoundaryAuxiliaryLoss,make_boundary_target
ROOT=Path(__file__).resolve().parents[1];SPL=ROOT/"data"/"PYCrackDB"/"splits";AN=ROOT/"analysis"/"PYCRACKDB_THIRD_DATASET_FINAL_VERIFICATION";LAMBDA_B=.10
SPLITS={"train":SPL/"train_seed2026.txt","val":SPL/"val_seed2026.txt","test":SPL/"test_seed2026.txt"};EXPECTED={"baseline":587263,"cssg":587458,"lbas":587651};THRESHOLDS=[x/100 for x in range(50,100)]
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
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=("train",),required=True);p.add_argument("--model",choices=("all","baseline","cssg","lbas"),required=True)
 p.add_argument("--data-root",default="data/PYCrackDB");p.add_argument("--output-dir");p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=2026);p.add_argument("--input-size",type=int,default=512);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--epochs",type=int,default=100);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--validation-interval",type=int,default=10);p.add_argument("--pretrained",type=as_bool,default=False);p.add_argument("--threshold-min",type=float,default=.50);p.add_argument("--threshold-max",type=float,default=.99);p.add_argument("--threshold-step",type=float,default=.01);p.add_argument("--lambda-b",type=float,default=.10);p.add_argument("--smoke-steps",type=int,default=5)
 a=p.parse_args()
 if (a.seed,a.input_size,a.batch_size,a.epochs,a.validation_interval,a.pretrained)!=(2026,512,16,100,10,False):p.error("locked protocol mismatch")
 if (a.threshold_min,a.threshold_max,a.threshold_step)!=(.50,.99,.01) or a.lambda_b!=.10:p.error("locked calibration/lambda mismatch")
 if a.mode=="smoke" and (a.model!="all" or a.smoke_steps!=5):p.error("smoke requires --model all --smoke-steps 5")
 if a.mode=="train" and (a.model=="all" or not a.output_dir):p.error("train requires one model and output-dir")
 return a
def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def seed_worker(_):
 s=torch.initial_seed()%2**32;random.seed(s);np.random.seed(s)
def loader(ds,b,w,shuffle,seed):return DataLoader(ds,batch_size=b,shuffle=shuffle,num_workers=w,pin_memory=True,drop_last=False,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed),persistent_workers=w>0)
def dataset(a,s):return PYCrackDBFixedDataset(a.data_root,Path(a.data_root)/"splits"/SPLITS[s].name,a.input_size,s=="train")
def build_model(kind,dev):
 stub_summary()
 if kind=="baseline":
  from models.RHANet_dep.rhanet_dep import RHANet;m=RHANet(object())
 elif kind=="cssg":
  from models.RHANet_CSSG.rhanet_cssg import RHANetCSSG;m=RHANetCSSG(object())
 else:m=CSSGBoundaryAuxTrainerModel(object())
 m=m.to(dev);total=sum(p.numel() for p in m.parameters());deploy=sum(p.numel() for p in (m.segmentation_model.parameters() if kind=="lbas" else m.parameters()))
 if total!=EXPECTED[kind]:raise RuntimeError(f"parameter mismatch {kind}: {total}")
 if kind=="lbas" and deploy!=EXPECTED["cssg"]:raise RuntimeError("deployment params mismatch")
 return m,total,deploy
def optim(m):
 o=torch.optim.Adam(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False);s=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode="min",factor=.95,patience=10,min_lr=1e-6,threshold=.0001,threshold_mode="rel",cooldown=0,eps=1e-8);return o,s
def counts(o,t,th):
 p=o>=th;g=t>=.5;return tuple(map(int,(torch.logical_and(p,g).sum(),torch.logical_and(p,~g).sum(),torch.logical_and(~p,~g).sum(),torch.logical_and(~p,g).sum())))
def basic(tp,fp,tn,fn):
 q=lambda a,b:float(a/b) if b else 0.;ci=q(tp,tp+fp+fn);bi=q(tn,tn+fp+fn);f=q(2*tp,2*tp+fp+fn);return {"accuracy":q(tp+tn,tp+fp+tn+fn),"precision":q(tp,tp+fp),"recall":q(tp,tp+fn),"f1":f,"crack_iou":ci,"background_iou":bi,"miou":(ci+bi)/2,"dice":f,"tp":tp,"fp":fp,"tn":tn,"fn":fn,"prediction_positive_ratio":q(tp+fp,tp+fp+tn+fn)}
def better(a,b):return b is None or (a["f1"],a["crack_iou"],-a["fp"])>(b["f1"],b["crack_iou"],-b["fp"])
def calibrate(o,t):
 best=None
 for th in THRESHOLDS:
  z={"threshold":th,**basic(*counts(o,t,th))}
  if better(z,best):best=z
 return best
def collect(m,dl,crit,dev,track_head=False):
 m.eval();oo=[];tt=[];loss=0.;n=0;calls=[0];handle=None
 if track_head:
  handle=m.boundary_head.register_forward_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
 try:
  with torch.no_grad():
   for im,t,_ in dl:
    im,t=im.to(dev,non_blocking=True),t.to(dev,non_blocking=True);o=m(im)
    if not torch.isfinite(o).all() or float(o.min())<0 or float(o.max())>1:raise RuntimeError("invalid probability")
    if crit is not None:
     z=crit(o,t)
     if not torch.isfinite(z):raise RuntimeError("nonfinite eval loss")
     loss+=float(z)
    oo.append(o.cpu());tt.append(t.cpu());n+=1
 finally:
  if handle:handle.remove()
 return torch.cat(oo),torch.cat(tt),(loss/n if crit is not None else None),calls[0]
def logger(out):
 out.mkdir(parents=True,exist_ok=False);x=logging.getLogger("pycrackdb_final");x.handlers.clear();x.setLevel(logging.INFO);fmt=logging.Formatter("%(asctime)s | %(message)s")
 for h in (logging.StreamHandler(),logging.FileHandler(out/"train.log",mode="w",encoding="utf-8")):h.setFormatter(fmt);x.addHandler(h)
 return x
def train(a):
 seed_all(a.seed);dev=torch.device(a.device);out=Path(a.output_dir);log=logger(out);td,vd,te=(dataset(a,s) for s in ("train","val","test"));tl=loader(td,a.batch_size,a.num_workers,True,a.seed);vl=loader(vd,a.batch_size,a.num_workers,False,a.seed);tel=loader(te,a.batch_size,a.num_workers,False,a.seed)
 pw=td.compute_pos_weight(3.);crit=WeightedBCELoss(pw,reduction="mean",from_logits=False).to(dev);aux=BoundaryAuxiliaryLoss().to(dev) if a.model=="lbas" else None;m,total,deploy=build_model(a.model,dev);op,sch=optim(m);log.info("start model=%s train=%d val=%d test=%d",a.model,len(td),len(vd),len(te))
 cfg=vars(a).copy();cfg.update({"dataset":"PY-CrackDB","protocol":"Fixed deterministic group-safe stratified 258/55/56 Seed-2026","split_files":{k:str(v) for k,v in SPLITS.items()},"model_name":{"baseline":"RHACrackNet*","cssg":"RHACrackNet* + CSSG","lbas":"RHACrackNet* + CSSG + LBAS"}[a.model],"params":total,"deployment_params":deploy,"optimizer":"Adam","lr":op.param_groups[0]["lr"],"betas":[.9,.999],"weight_decay":0.,"scheduler":"ReduceLROnPlateau(mode=min,factor=.95,patience=10,min_lr=1e-6)","warmup":"none","segmentation_loss":"WeightedBCELoss(from_logits=False)","pos_weight":pw,"augmentation":"Locked common H/V/180 flips + brightness 1.5/0.5 + contrast 1.5","normalization":"RGB / 255","image_resize":"bilinear 512x512","mask_binary":"PIL convert('1')","mask_resize":"nearest","checkpoint_selection":"highest calibrated Val strict crack F1; tie higher Crack IoU; tie lower FP","test_threshold":"frozen best_val_threshold","lambda_b":.10 if a.model=="lbas" else None,"boundary_target":"3x3 morphological gradient" if a.model=="lbas" else None,"boundary_loss":"0.5 BCE + 0.5 Dice" if a.model=="lbas" else None,"boundary_head_deployment":"off" if a.model=="lbas" else None})
 (out/"run_config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
 fields=["epoch","train_seg_loss","train_boundary_loss","train_total_loss","learning_rate","val_loss","val_threshold","val_precision","val_recall","val_f1","val_crack_iou","val_miou","val_boundary_f1_tol2","val_fp","val_fn","best_f1","best_epoch","best_val_threshold"];vrows=[];best=None;be=0;bt=None;start=time.perf_counter()
 with (out/"training_summary.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for ep in range(1,a.epochs+1):
   m.train();ss=bb=tot=0.
   for im,t,_ in tl:
    im,t=im.to(dev,non_blocking=True),t.to(dev,non_blocking=True);op.zero_grad(set_to_none=True)
    if a.model=="lbas":seg,boundary,_,_=m(im,return_aux=True);ls=crit(seg,t);lb,_=aux(boundary,make_boundary_target(t));z=ls+.10*lb
    else:ls=crit(m(im),t);lb=torch.zeros((),device=dev);z=ls
    if not torch.isfinite(z):raise RuntimeError("nonfinite training loss")
    z.backward();op.step();ss+=float(ls);bb+=float(lb);tot+=float(z)
   row={k:"" for k in fields};row.update({"epoch":ep,"train_seg_loss":ss/len(tl),"train_boundary_loss":bb/len(tl) if a.model=="lbas" else "","train_total_loss":tot/len(tl),"learning_rate":op.param_groups[0]["lr"],"best_f1":best["f1"] if best else -1.,"best_epoch":be,"best_val_threshold":bt})
   if ep%a.validation_interval==0:
    o,t,vloss,_=collect(m,vl,crit,dev);v=calibrate(o,t);full=evaluate(o,t,v["threshold"],(2,));sch.step(vloss);vr={"epoch":ep,**v,"boundary_f1_tol2":full["boundary_f1_tol2"]};vrows.append(vr)
    if better(v,best):best,be,bt=v,ep,v["threshold"];torch.save({"model_state_dict":m.state_dict(),"model_variant":a.model,"best_epoch":be,"best_val_f1":best["f1"],"best_val_threshold":bt},out/"best_model.pth")
    row.update({"val_loss":vloss,"val_threshold":v["threshold"],"val_precision":v["precision"],"val_recall":v["recall"],"val_f1":v["f1"],"val_crack_iou":v["crack_iou"],"val_miou":v["miou"],"val_boundary_f1_tol2":full["boundary_f1_tol2"],"val_fp":v["fp"],"val_fn":v["fn"],"best_f1":best["f1"],"best_epoch":be,"best_val_threshold":bt});log.info("epoch %d val %s",ep,json.dumps(v))
   w.writerow(row);f.flush();log.info("epoch %d/%d total %.6f best %.6f@%d T=%s",ep,a.epochs,row["train_total_loss"],best["f1"] if best else -1.,be,bt)
 with (out/"validation_epoch_metrics.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(vrows[0]));w.writeheader();w.writerows(vrows)
 p=torch.load(out/"best_model.pth",map_location=dev);m.load_state_dict(p["model_state_dict"],strict=True);vp,vt,_,_=collect(m,vl,None,dev);re=calibrate(vp,vt)
 if re["threshold"]!=bt:raise RuntimeError("frozen threshold verification failed")
 tp,tt,_,calls=collect(m,tel,None,dev,a.model=="lbas");val={**re,**evaluate(vp,vt,bt,(2,5,8))};test={"dataset":"PY-CrackDB","split":"test","model":a.model,"threshold_source":"frozen Val threshold","threshold":bt,**evaluate(tp,tt,bt,(2,5,8)),"reference_0_5":evaluate(tp,tt,.5,(2,5,8)),"params":total,"deployment_params":deploy,"boundary_head_called":bool(calls) if a.model=="lbas" else None,"best_epoch":be}
 if a.model=="lbas" and calls:raise RuntimeError("boundary head called during Test")
 final={"status":"COMPLETE","dataset":"PY-CrackDB","model":a.model,"seed":a.seed,"input_size":a.input_size,"batch_size":a.batch_size,"epochs":a.epochs,"best_epoch":be,"best_val_threshold":bt,"best_val_f1":best["f1"],"val":val,"test":test,"total_training_time_sec":time.perf_counter()-start};(out/"final_metrics.json").write_text(json.dumps(final,indent=2),encoding="utf-8");print(json.dumps(final,indent=2))
def clean(m=None):
 if m is not None:del m
 gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()
def main():
 a=arguments()
 train(a)
 clean()
if __name__=="__main__":main()