"""PY-CrackDB fixed-split adapter using the locked common preprocessing protocol."""
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image,ImageEnhance,ImageOps
from torch.utils.data import Dataset

class PYCrackDBFixedDataset(Dataset):
    def __init__(self,data_root,split_file,input_size=512,train=False):
        self.root=Path(data_root).resolve();self.split_file=Path(split_file).resolve()
        self.input_size=int(input_size);self.train=bool(train);self.samples=[]
        if not self.split_file.is_file():raise FileNotFoundError(self.split_file)
        for n,line in enumerate(self.split_file.read_text(encoding="utf-8-sig").splitlines(),1):
            if not line.strip():continue
            q=line.split("\t")
            if len(q)!=2:raise ValueError(f"bad split line {n}: {line}")
            ip=(self.root/q[0]).resolve();mp=(self.root/q[1]).resolve()
            if self.root not in ip.parents or self.root not in mp.parents:raise ValueError("path escapes data root")
            if not ip.is_file() or not mp.is_file():raise FileNotFoundError(f"{ip}|{mp}")
            self.samples.append((ip,mp,q[0].replace("\\","/")))
        if not self.samples:raise RuntimeError("empty split")
    def __len__(self):return len(self.samples)
    @staticmethod
    def augment_pair(image,mask):
        if random.random()>.5:image,mask=ImageOps.mirror(image),ImageOps.mirror(mask)
        if random.random()>.5:image,mask=ImageOps.flip(image),ImageOps.flip(mask)
        if random.random()>.5:
            image=ImageOps.flip(ImageOps.mirror(image));mask=ImageOps.flip(ImageOps.mirror(mask))
        if random.random()>.5:image=ImageEnhance.Brightness(image).enhance(1.5)
        if random.random()>.5:image=ImageEnhance.Brightness(image).enhance(.5)
        if random.random()>.5:image=ImageEnhance.Contrast(image).enhance(1.5)
        return image,mask
    def __getitem__(self,index):
        ip,mp,sid=self.samples[index]
        with Image.open(ip) as x:image=x.convert("RGB")
        with Image.open(mp) as x:mask=x.convert("1")
        size=(self.input_size,self.input_size)
        image=image.resize(size,Image.Resampling.BILINEAR)
        mask=mask.resize(size,Image.Resampling.NEAREST)
        if self.train:image,mask=self.augment_pair(image,mask)
        ia=np.asarray(image,dtype=np.float32)/255.
        ma=np.asarray(mask,dtype=np.uint8).astype(np.float32)
        if not set(np.unique(ma)).issubset({0.,1.}):raise RuntimeError(f"nonbinary mask: {mp}")
        return torch.from_numpy(ia.transpose(2,0,1).copy()),torch.from_numpy(ma[None].copy()),sid
    def compute_pos_weight(self,factor=3.):
        pos=total=0;size=(self.input_size,self.input_size)
        for _,mp,_ in self.samples:
            with Image.open(mp) as x:m=np.asarray(x.convert("1").resize(size,Image.Resampling.NEAREST),dtype=bool)
            pos+=int(m.sum());total+=m.size
        if not pos:raise RuntimeError("no positive pixels")
        return total/pos*float(factor)

