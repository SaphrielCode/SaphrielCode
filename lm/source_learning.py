"""Response-only per-example objectives and provenance-aware weights."""
from collections import Counter
import torch
from torch.nn import functional as F

def family_weights(rows):
 counts=Counter(e['family'] for e in rows);n=len(rows);g=len(counts)
 return torch.tensor([n/(g*counts[e['family']]) for e in rows],dtype=torch.float32)
def encode(rows,tokenizer,context=128):
 xs=[];ys=[]
 for e in rows:
  pi=tokenizer.encode('Instruction:\n'+e['instruction']+'\nResponse:\n').ids;ri=tokenizer.encode(e['response']).ids;seq=[1]+pi+ri+[2];x=seq[:-1];y=seq[1:];y[:len(pi)]=[-100]*len(pi)
  if len(x)>context:raise ValueError('Instruction exceeds frozen context: '+e['id'])
  xs.append(x+[0]*(context-len(x)));ys.append(y+[-100]*(context-len(y)))
 return torch.tensor(xs,dtype=torch.long),torch.tensor(ys,dtype=torch.long)
def response_loss(model,x,y,weights=None):
 # Right padding only; prompt positions remain context, but are never supervised.
 length=int(x.ne(0).any(0).nonzero()[-1])+1;x=x[:,:length];y=y[:,:length];valid=y.ne(-100);counts=valid.sum(1)
 if (counts==0).any():raise ValueError('An example has zero response targets')
 logits=model(x,valid=valid);ce=F.cross_entropy(logits,y[valid],reduction='none');owners=valid.nonzero()[:,0];sums=torch.zeros(len(x),dtype=ce.dtype).scatter_add(0,owners,ce);per_example=sums/counts
 loss=(per_example if weights is None else per_example*weights).mean()
 return loss,ce.detach().double().sum(),int(valid.sum())
