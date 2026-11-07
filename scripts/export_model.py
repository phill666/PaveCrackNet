import argparse, torch
from runtime import load_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
 m,payload=load_model('lbas',a.checkpoint,a.device)
 torch.save({'model_state_dict':m.segmentation_model.state_dict(),'model_variant':'locked_cssg_deployment','best_val_threshold':payload.get('best_val_threshold')},a.output)
if __name__=='__main__':main()
