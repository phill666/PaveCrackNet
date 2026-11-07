"""Formal defaults; explicit custom opt-in for non-paper runs."""
import argparse, importlib, pathlib, sys
from types import SimpleNamespace
import yaml
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--allow-custom',action='store_true');a=p.parse_args()
 c=yaml.safe_load(pathlib.Path(a.config).read_text())
 fixed={'batch_size':16,'epochs':100,'learning_rate':.001,'seed':2026,'input_size':512,'validation_interval':10}
 if not a.allow_custom:
  for k,v in fixed.items():
   if c[k]!=v:raise ValueError(f'{k} must be {v}; custom runs require --allow-custom and are not paper reproduction')
 ds=c['dataset_name'];model=c['model_name']
 if model not in ('baseline','cssg','lbas'):raise ValueError('model_name: baseline/cssg/lbas')
 name=({'baseline':'tut_baseline_protocolfix','cssg':'tut_cssg_protocolfix','lbas':'tut_cssg_lbas'}[model] if ds=='tut' else {'camcrack789':'camcrack789_experiment','pycrackdb':'pycrackdb_final_experiment'}[ds])
 m=importlib.import_module(name)
 args=SimpleNamespace(**{k:c[k] for k in ('device','batch_size','epochs','seed','input_size','num_workers','validation_interval')},data_root=c['dataset_root'],output_dir=c['output_dir'],mode='train',model=model,pretrained=False,lambda_b=.1,threshold_min=.5,threshold_max=.99,threshold_step=.01,checkpoint=None,smoke_steps=5)
 if a.allow_custom:
  key='optimizer_scheduler' if name=='tut_cssg_lbas' else 'optim'
  original=getattr(m,key)
  def custom_optimizer(model):
   op,sch=original(model)
   for group in op.param_groups:group['lr']=float(c['learning_rate'])
   return op,sch
  setattr(m,key,custom_optimizer)
  print('CUSTOM RUN: not the locked paper protocol')
 if ds=='tut' and model!='lbas':
  m.args=lambda:args;m.main()
 else:m.train(args)
if __name__=='__main__':main()
