"""Opt-in efficiency experiments. Existing checkpoint layout and baseline stay unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from lm.experimental_model import Decoder

class EfficientDecoder(Decoder):
    def __init__(self, cfg, seed=42, attention='manual', factor_dim=None):
        super().__init__(cfg, seed)
        if attention not in ('manual','sdpa'): raise ValueError(attention)
        self.attention = attention
        self.factor_dim = factor_dim
        if factor_dim:
            self.embedding = nn.Embedding(cfg.vocab_size, factor_dim)
            self.token_projection = nn.Parameter(torch.empty(cfg.width, factor_dim))
            g = torch.Generator().manual_seed(seed + 1000)
            with torch.no_grad():
                nn.init.normal_(self.embedding.weight, std=.02, generator=g)
                nn.init.normal_(self.token_projection, std=1/math.sqrt(factor_dim), generator=g)

    def embed(self, ids):
        x = self.embedding(ids)
        return F.linear(x, self.token_projection) if self.factor_dim else x

    def project(self, hidden):
        if self.factor_dim: hidden = F.linear(hidden, self.token_projection.T)
        return F.linear(hidden, self.embedding.weight)

    def hidden(self, ids, cache=None):
        if ids.ndim != 2 or not ids.shape[1]: raise ValueError('Expected nonempty [batch, length]')
        offset = 0 if cache is None else cache[0][0].shape[2]
        length = ids.shape[1]
        if offset+length > self.config.max_context: raise ValueError('Context overflow')
        x = self.embed(ids); new_cache=[]
        cos=self.rope_cos[offset:offset+length];sin=self.rope_sin[offset:offset+length]
        for i,b in enumerate(self.blocks):
            q,k,v=b.qkv(b.attn_norm(x)).chunk(3,-1)
            q,k,v=[t.reshape(ids.shape[0],length,b.heads,b.head_dim).transpose(1,2) for t in (q,k,v)]
            q,k=b.rotate(q,cos,sin),b.rotate(k,cos,sin)
            if cache is not None:
                k=torch.cat((cache[i][0],k),2);v=torch.cat((cache[i][1],v),2)
            new_cache.append((k,v))
            mask=self.causal_mask[offset:offset+length,:offset+length]
            if self.attention=='sdpa':
                attended=F.scaled_dot_product_attention(q,k,v,attn_mask=~mask,dropout_p=0.)
            else:
                scores=(q@k.transpose(-2,-1))/math.sqrt(b.head_dim)
                attended=scores.masked_fill(mask,float('-inf')).softmax(-1)@v
            x=x+b.attn_out(attended.transpose(1,2).contiguous().reshape_as(x))
            n=b.ff_norm(x);h=b.ff_in(n)
            h=F.silu(b.ff_gate(n))*h if b.ff_activation=='swiglu' else F.gelu(h)
            x=x+b.ff_out(h)
        return self.final_norm(x),new_cache

    def forward(self, ids, valid=None, last_only=False):
        hidden,_=self.hidden(ids)
        if valid is not None: hidden=hidden[valid]
        elif last_only: hidden=hidden[:,-1,:]
        return self.project(hidden)

    @torch.no_grad()
    def cached(self, ids, cache=None):
        if self.training: raise ValueError('Cache is inference only')
        hidden,cache=self.hidden(ids,cache)
        return self.project(hidden[:,-1,:]),cache

class RollingCache:
    """Rebuild on overflow; dropping KV alone changes multi-layer conditioning."""
    def __init__(self, model, context=128):
        self.model=model;self.context=context;self.cache=None;self.previous=None;self.rebuilds=0
    @torch.no_grad()
    def __call__(self, ids):
        window=ids[:,-self.context:]
        incremental=(self.previous is not None and ids.shape[1]<=self.context and
                     ids.shape[1]==self.previous.shape[1]+1 and torch.equal(ids[:,:-1],self.previous))
        if incremental: scores,self.cache=self.model.cached(ids[:,-1:],self.cache)
        else:
            scores,self.cache=self.model.cached(window);self.rebuilds+=1
        self.previous=ids.clone()
        return scores

def training_loss(model,x,y,valid_only=False,trim=False):
    if not y.ne(-100).any(): raise ValueError('No valid targets')
    if trim:
        length=int(y.ne(-100).any(0).nonzero()[-1])+1
        x,y=x[:,:length],y[:,:length]
    valid=y.ne(-100);n=int(valid.sum())
    if not n: raise ValueError('No valid targets')
    logits=model(x,valid=valid) if valid_only else model(x)
    targets=y[valid] if valid_only else y.flatten()
    losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets,reduction='none',ignore_index=-100)
    return losses.sum()/n,losses.detach().double().sum(),n
