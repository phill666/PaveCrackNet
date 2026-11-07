"""TUT locked CSSG with training-only LBAS; formal training requires --mode train."""
from __future__ import annotations
import argparse,csv,gc,json,logging,random,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from dataset.TUT_dataset import TUTDataset
from losses.cross_entropy import WeightedBCELoss
from evaluation_utils import calibrate,collect,evaluate,cnt,metrics
from models.RHANet_CSSG_LBAS import CSSGBoundaryAuxTrainerModel,BoundaryAuxiliaryLoss,make_boundary_target

CSSG_PARAMS=587458
HEAD_PARAMS=193
TRAINING_PARAMS=587651
CSSG_MACS=4101437728
HEAD_MACS=10485760
TRAINING_MACS=CSSG_MACS+HEAD_MACS
LAMBDA_B=.10

def as_bool(v):
    if isinstance(v,bool):return v
    if v.lower() in {"false","0","no"}:return False
    if v.lower() in {"true","1","yes"}:return True
    raise argparse.ArgumentTypeError("expected true/false")

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("train","val","test","complexity"),required=True)
    p.add_argument("--data-root",default="data/TUT")
    p.add_argument("--output-dir",default="outputs/TUT_PaveCrackNet")
    p.add_argument("--checkpoint")
    p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--input-size",type=int,default=512);p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--batch-size",type=int,default=16);p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--validation-interval",type=int,default=10);p.add_argument("--pretrained",type=as_bool,default=False)
    p.add_argument("--lambda-b",type=float,default=.10)
    p.add_argument("--threshold-min",type=float,default=.50);p.add_argument("--threshold-max",type=float,default=.99)
    p.add_argument("--threshold-step",type=float,default=.01);p.add_argument("--smoke-steps",type=int,default=5)
    a=p.parse_args()
    if a.lambda_b!=.10:p.error("lambda_b is fixed at 0.10")
    if (a.threshold_min,a.threshold_max,a.threshold_step)!=(.50,.99,.01):p.error("threshold protocol fixed at .50:.01:.99")
    if a.pretrained:p.error("pretrained must remain false")
    if a.mode in {"val","test"} and not a.checkpoint:p.error("val/test require --checkpoint")
    return a

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def seed_worker(_):
    s=torch.initial_seed()%(2**32);random.seed(s);np.random.seed(s)

def loader(ds,batch,workers,shuffle,seed,drop_last=False):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,
        drop_last=drop_last,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed),
        persistent_workers=workers>0)

def build_model(device):
    m=CSSGBoundaryAuxTrainerModel(object()).to(device)
    total=sum(p.numel() for p in m.parameters());head=sum(p.numel() for p in m.boundary_head.parameters())
    deploy=sum(p.numel() for p in m.segmentation_model.parameters())
    if (total,head,deploy)!=(TRAINING_PARAMS,HEAD_PARAMS,CSSG_PARAMS):
        raise RuntimeError(f"parameter mismatch {total}/{head}/{deploy}")
    return m,total,head,deploy

def optimizer_scheduler(model):
    op=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(op,mode="min",factor=.95,patience=10,min_lr=1e-6,
        threshold=.0001,threshold_mode="rel",cooldown=0,eps=1e-8)
    return op,sch

def strict_load(model,path,device):
    p=torch.load(path,map_location=device)
    x=model.load_state_dict(p["model_state_dict"],strict=True)
    if x.missing_keys or x.unexpected_keys:raise RuntimeError(str(x))
    return p

def logger(out):
    out.mkdir(parents=True,exist_ok=True);x=logging.getLogger("tut_cssg_lbas");x.handlers.clear();x.setLevel(logging.INFO)
    fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(),logging.FileHandler(out/"train.log",mode="w",encoding="utf-8")):
        h.setFormatter(fmt);x.addHandler(h)
    return x

def aux_counts(logits,target):
    return metrics(*cnt(torch.sigmoid(logits),target,.5))

def validate(model,dl,criterion,aux_loss,device):
    model.eval();oo=[];tt=[];losses=[];aux_total=[0,0,0,0]
    with torch.no_grad():
        for images,target in dl:
            images,target=images.to(device,non_blocking=True),target.to(device,non_blocking=True)
            seg,boundary,_,_=model(images,return_aux=True);bt=make_boundary_target(target)
            lb,_=aux_loss(boundary,bt);ls=criterion(seg,target);losses.append(float(ls))
            oo.append(seg.cpu());tt.append(target.cpu())
            c=cnt(torch.sigmoid(boundary).cpu(),bt.cpu(),.5);aux_total=[a+b for a,b in zip(aux_total,c)]
    return torch.cat(oo),torch.cat(tt),sum(losses)/len(losses),metrics(*aux_total)

def train(a):
    seed_all(a.seed);dev=torch.device(a.device);out=Path(a.output_dir);log=logger(out)
    train_set=TUTDataset(a.data_root,"train",a.input_size,True);val_set=TUTDataset(a.data_root,"val",a.input_size,False)
    tl=loader(train_set,a.batch_size,a.num_workers,True,a.seed,True);vl=loader(val_set,a.batch_size,a.num_workers,False,a.seed)
    seg_loss=WeightedBCELoss(train_set.compute_pos_weight(3.0),reduction="mean",from_logits=False).to(dev)
    boundary_loss=BoundaryAuxiliaryLoss().to(dev);model,total,head,deploy=build_model(dev);op,sch=optimizer_scheduler(model)
    cfg=vars(a).copy();cfg.update({"dataset":"TUT","model":"RHACrackNet* + locked CSSG + training-only LBAS",
        "class":type(model).__name__,"source":"code/models/RHANet_CSSG_LBAS/lbas.py",
        "training_params":total,"boundary_head_params":head,"deployment_params":deploy,
        "training_macs":TRAINING_MACS,"deployment_macs":CSSG_MACS,"optimizer":"Adam","lr":op.param_groups[0]["lr"],
        "weight_decay":0.,"scheduler":"ReduceLROnPlateau(factor=.95,patience=10,min_lr=1e-6)",
        "segmentation_loss":"WeightedBCELoss(from_logits=False)","boundary_loss":"0.5*BCEWithLogits+0.5*SoftDice",
        "boundary_target":"3x3 morphological gradient","lambda_b":LAMBDA_B,
        "augmentation":"locked TUT CSSG V1","normalization":"RGB / 255","mask_resize":"nearest",
        "checkpoint_selection":"highest calibrated Val strict crack F1","test_threshold":"frozen best_val_threshold",
        "boundary_head_deployment":"discarded"})
    (out/"run_config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    fields=["epoch","train_seg_loss","train_boundary_loss","train_total_loss","learning_rate","val_loss",
        "val_threshold","val_precision","val_recall","val_f1","val_crack_iou","val_miou",
        "aux_boundary_precision","aux_boundary_recall","aux_boundary_f1","best_f1","best_epoch","best_val_threshold"]
    best=-1.;best_epoch=0;best_th=None;started=time.perf_counter()
    with (out/"training_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for epoch in range(1,a.epochs+1):
            model.train();ss=bb=tt=0.
            for images,target in tl:
                images,target=images.to(dev,non_blocking=True),target.to(dev,non_blocking=True)
                op.zero_grad(set_to_none=True);seg,boundary,_,_=model(images,return_aux=True)
                ls=seg_loss(seg,target);bt=make_boundary_target(target);lb,_=boundary_loss(boundary,bt);total_loss=ls+LAMBDA_B*lb
                if not torch.isfinite(total_loss):raise FloatingPointError("non-finite training loss")
                total_loss.backward();op.step();ss+=float(ls);bb+=float(lb);tt+=float(total_loss)
            row={k:"" for k in fields};row.update({"epoch":epoch,"train_seg_loss":ss/len(tl),
                "train_boundary_loss":bb/len(tl),"train_total_loss":tt/len(tl),"learning_rate":op.param_groups[0]["lr"],
                "best_f1":best,"best_epoch":best_epoch,"best_val_threshold":best_th})
            if epoch%a.validation_interval==0:
                prob,target,vloss,aux=validate(model,vl,seg_loss,boundary_loss,dev);v,_=calibrate(prob,target);sch.step(vloss)
                if v["f1"]>best:
                    best,best_epoch,best_th=v["f1"],epoch,v["threshold"]
                    torch.save({"model_state_dict":model.state_dict(),"model_variant":"cssg_lbas",
                        "best_epoch":best_epoch,"best_val_f1":best,"best_val_threshold":best_th},out/"best_model.pth")
                row.update({"val_loss":vloss,"val_threshold":v["threshold"],"val_precision":v["precision"],
                    "val_recall":v["recall"],"val_f1":v["f1"],"val_crack_iou":v["crack_iou"],"val_miou":v["miou"],
                    "aux_boundary_precision":aux["precision"],"aux_boundary_recall":aux["recall"],"aux_boundary_f1":aux["f1"],
                    "best_f1":best,"best_epoch":best_epoch,"best_val_threshold":best_th})
                log.info("epoch %d calibrated val %s aux %s",epoch,json.dumps(v),json.dumps(aux))
            w.writerow(row);f.flush();log.info("epoch %d/%d total %.6f best %.6f@%d T=%s",epoch,a.epochs,row["train_total_loss"],best,best_epoch,best_th)
    (out/"final_metrics.json").write_text(json.dumps({"status":"TRAINING_COMPLETE_AWAITING_EXPLICIT_VAL_TEST",
        "dataset":"TUT","seed":a.seed,"input_size":a.input_size,"batch_size":a.batch_size,"epochs":a.epochs,
        "best_epoch":best_epoch,"best_val_threshold":best_th,"best_val_f1":best,
        "total_training_time_sec":time.perf_counter()-started},indent=2),encoding="utf-8")

def run_eval(a):
    seed_all(a.seed);dev=torch.device(a.device);split="val" if a.mode=="val" else "test"
    ds=TUTDataset(a.data_root,split,a.input_size,False);dl=loader(ds,a.batch_size,a.num_workers,False,a.seed)
    model,total,_,deploy=build_model(dev);payload=strict_load(model,a.checkpoint,dev)
    probabilities,targets,_=collect(model,dl,None,dev)
    if split=="val":
        best,_=calibrate(probabilities,targets);result={"dataset":"TUT","split":"val","threshold_source":"Val calibration",
            **best,**evaluate(probabilities,targets,best["threshold"],(2,5,8)),"training_params":total,"deployment_params":deploy}
        if payload.get("best_val_threshold") is not None and best["threshold"]!=payload["best_val_threshold"]:
            raise RuntimeError("recomputed Val threshold differs from checkpoint")
    else:
        th=payload.get("best_val_threshold")
        if th is None:raise RuntimeError("checkpoint lacks frozen best_val_threshold")
        result={"dataset":"TUT","split":"test","threshold_source":"frozen Val threshold","threshold":th,
            **evaluate(probabilities,targets,th,(2,5,8)),"training_params":total,"deployment_params":deploy,
            "boundary_head_called":False,"best_epoch":payload.get("best_epoch"),"best_val_f1":payload.get("best_val_f1")}
    print(json.dumps(result,indent=2));path=Path(a.output_dir)/"final_metrics.json"
    current=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {};current[split]=result
    path.write_text(json.dumps(current,indent=2),encoding="utf-8")

def complexity(a):
    _,total,head,deploy=build_model(torch.device(a.device))
    print(json.dumps({"cssg_params":CSSG_PARAMS,"boundary_head_params":head,"training_params":total,
        "training_delta_percent":100*head/CSSG_PARAMS,"deployment_params":deploy,"deployment_delta_params":0,
        "cssg_macs":CSSG_MACS,"training_macs":TRAINING_MACS,"training_delta_macs":HEAD_MACS,
        "training_delta_macs_percent":100*HEAD_MACS/CSSG_MACS,"deployment_macs":CSSG_MACS,
        "deployment_delta_macs":0},indent=2))


def main():
    a=arguments()
    if a.mode=="train":train(a)
    elif a.mode in {"val","test"}:run_eval(a)
    else:complexity(a)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
if __name__=="__main__":main()
