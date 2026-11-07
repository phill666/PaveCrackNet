import torch, cv2
import numpy as np
THRESHOLDS=[x/100 for x in range(50,100)]

def cnt(o,t,th):
 p=o>=th;g=t>=.5
 return tuple(map(int,(torch.logical_and(p,g).sum(),torch.logical_and(p,~g).sum(),torch.logical_and(~p,~g).sum(),torch.logical_and(~p,g).sum())))

def metrics(tp,fp,tn,fn):
 q=lambda n,d:float(n/d) if d else 0.;ci=q(tp,tp+fp+fn);bi=q(tn,tn+fp+fn);f=q(2*tp,2*tp+fp+fn)
 return {"accuracy":q(tp+tn,tp+fp+tn+fn),"precision":q(tp,tp+fp),"recall":q(tp,tp+fn),"f1":f,"crack_iou":ci,"background_iou":bi,"miou":(ci+bi)/2,"dice":f,"tp":tp,"fp":fp,"tn":tn,"fn":fn,"prediction_positive_ratio":q(tp+fp,tp+fp+tn+fn)}

def collect(m,dl,criterion,device,max_batches=None):
 m.eval();oo=[];tt=[];loss=0.;n=0
 with torch.no_grad():
  for im,t in dl:
   im,t=im.to(device,non_blocking=True),t.to(device,non_blocking=True);o=m(im)
   if not torch.isfinite(o).all() or float(o.min())<0 or float(o.max())>1:raise RuntimeError("non-probability/non-finite output")
   if criterion is not None:
    z=criterion(o,t)
    if not torch.isfinite(z):raise RuntimeError("non-finite eval loss")
    loss+=float(z)
   oo.append(o.cpu());tt.append(t.cpu());n+=1
   if max_batches and n>=max_batches:break
 return torch.cat(oo),torch.cat(tt),(loss/n if criterion is not None else None)

def calibrate(o,t):
 rows=[];best=None
 for th in THRESHOLDS:
  r={"threshold":th,**metrics(*cnt(o,t,th))};rows.append(r)
  if best is None or r["f1"]>best["f1"] or (r["f1"]==best["f1"] and th>best["threshold"]):best=r
 return best,rows

def bcounts(o,t,th,tol):
 ps=(o.numpy()[:,0]>=th).astype(np.uint8);gs=(t.numpy()[:,0]>=.5).astype(np.uint8);k=np.ones((3,3),np.uint8);kd=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tol+1,2*tol+1))
 z={x:0 for x in ("pb","gb","pm","gm","near","fp","bfn","fn")}
 for p,g in zip(ps,gs):
  gb=g-cv2.erode(g,k);pb=p-cv2.erode(p,k);gd=cv2.dilate(gb,kd);pd=cv2.dilate(pb,kd)
  z["pb"]+=int(pb.sum());z["gb"]+=int(gb.sum());z["pm"]+=int(((pb>0)&(gd>0)).sum());z["gm"]+=int(((gb>0)&(pd>0)).sum())
  fp=(p>0)&(g==0);fn=(p==0)&(g>0);dist=cv2.distanceTransform((g==0).astype(np.uint8),cv2.DIST_L2,5) if g.any() else np.full(g.shape,999.,np.float32)
  z["near"]+=int((fp&(dist<=3)).sum());z["fp"]+=int(fp.sum());z["bfn"]+=int((fn&(gd>0)).sum());z["fn"]+=int(fn.sum())
 return z

def bmetrics(z):
 q=lambda n,d:float(n/d) if d else 0.;p=q(z["pm"],z["pb"]);r=q(z["gm"],z["gb"])
 return {"boundary_precision":p,"boundary_recall":r,"boundary_f1":q(2*p*r,p+r),"near_boundary_fp":z["near"],"near_boundary_fp_ratio":q(z["near"],z["fp"]),"boundary_fn":z["bfn"],"boundary_fn_ratio":q(z["bfn"],z["fn"])}

def evaluate(o,t,th,tols=()):
 r=metrics(*cnt(o,t,th))
 for tol in tols:
  for k,v in bmetrics(bcounts(o,t,th,tol)).items():r[k+f"_tol{tol}"]=v
 if 2 in tols:
  for k in ("boundary_precision","boundary_recall","boundary_f1","near_boundary_fp","near_boundary_fp_ratio","boundary_fn","boundary_fn_ratio"):r[k]=r[k+"_tol2"]
 return r
