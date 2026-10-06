"""Arena training execution paths; exported weights and inference stay native.

Zero-threshold Delta cells have dense equivalents. Their temporary GRU gates
are derived tensors, never extra learned parameters. Evaluation always calls
the original backbone, including its deployed arithmetic and state semantics.
"""
from __future__ import annotations

import os
from types import MethodType

import torch
from torch import nn

_COMPILED = None
COMPILED_KEYS = frozenset({'pgjanet', 'dvrjanet', 'apnrru'})


def _dense_janet(layer, x):
    # h=(c+1)/2 turns sigmoid(g) into (tanh(g/2)+1)/2. The
    # corresponding dense GRU starts at c=-1 and has reset gate exactly one.
    wf, wg = layer.weight_ih_l0.chunk(2, 0)
    uf, ug = layer.weight_hh_l0.chunk(2, 0)
    bf, bg = (layer.bias_ih_l0 + layer.bias_hh_l0).chunk(2, 0)
    wx = torch.cat((torch.zeros_like(wf), wf, .5*wg), 0)
    wh = torch.cat((torch.zeros_like(uf), .5*uf, .25*ug), 0)
    bias = torch.cat((torch.full_like(bf, 80.), bf + .5*uf.sum(1), .5*bg + .25*ug.sum(1)), 0)
    initial = x.new_full((1, x.shape[0], layer.hidden_size), -1.)
    output, _ = torch._VF.gru(x, initial, [wx, wh, bias, torch.zeros_like(bias)],
                              True, 1, 0., True, False, True)
    return .5 * (output + 1.)


def _dense_delta_forward(self, x, *states):
    if (self.training and torch.is_grad_enabled() and not any(v is not None for v in states)
            and self.th_x == self.th_h == 0 and not self.debug
            and self.num_layers == 1 and x.dtype in (torch.float32, torch.float64)
            and os.getenv('OPENDPD_DISABLE_ARENA_DENSE', '0') != '1'):
        if isinstance(self, nn.GRU):
            return nn.GRU.forward(self, x, None)[0]
        if hasattr(self, 'x2h'):  # TRes-DeltaGRU has bias-free linear gates.
            initial=x.new_zeros((1,x.shape[0],self.hidden_size))
            return torch._VF.gru(x,initial,[self.x2h.weight,self.h2h.weight],
                                 False,1,0.,True,False,True)[0]
        return _dense_janet(self, x)
    return self._arena_eager_forward(x, *states)


def dense_training_module(module):
    return (getattr(getattr(module, 'forward', None), '__func__', None) is _dense_delta_forward
            and module.th_x == module.th_h == 0 and not module.debug and module.num_layers == 1)


def _dense_bojanet_forward(self, x, h_0):
    if not (self.training and torch.is_grad_enabled() and x.dtype in (torch.float32,torch.float64)
            and os.getenv('OPENDPD_DISABLE_ARENA_DENSE','0')!='1'):
        return self._arena_eager_forward(x,h_0)
    batch,length,_=x.shape
    pad=torch.zeros_like(x[:,-(self.window_size-1):,:])
    windows=torch.cat((pad,x),1).unfold(1,self.window_size,1).transpose(2,3)
    i=self.fir_I(windows[:,:,:,0])-self.fir_Q(windows[:,:,:,1])
    q=self.fir_Q(windows[:,:,:,0])+self.fir_I(windows[:,:,:,1])
    magnitude,squared,sine,cosine=self.vd_module(i,q)
    features=torch.stack((magnitude,squared),2).reshape(batch,length,self.num_vd_units*2)
    wf,wg=self.W_fi.weight,self.W_gi.weight
    uf,ug=self.W_fh.weight,self.W_gh.weight
    bf=self.W_fi.bias if self.W_fi.bias is not None else wf.new_zeros(self.hidden_size)
    bg=self.W_gi.bias if self.W_gi.bias is not None else wg.new_zeros(self.hidden_size)
    wx=torch.cat((torch.zeros_like(wf),wf,wg),0)
    wh=torch.cat((torch.zeros_like(uf),uf,ug),0)
    bias=torch.cat((torch.full_like(bf,80.),bf,bg),0)
    initial=x.new_zeros((1,batch,self.hidden_size)) if h_0 is None else h_0.reshape(1,batch,self.hidden_size)
    hidden=torch._VF.gru(features,initial,[wx,wh,bias,torch.zeros_like(bias)],True,1,0.,True,False,True)[0]
    i_rot,q_rot=self.pr_block(hidden,sine,cosine,self.hidden_size,self.num_vd_units)
    return torch.cat((self.W_out_I(i_rot)-self.W_out_Q(q_rot),
                      self.W_out_Q(q_rot)+self.W_out_I(i_rot)),-1)


def _compiled_forward(self, x, h_0=None):
    global _COMPILED
    if (self.training and torch.is_grad_enabled() and x.is_cuda and x.dtype == torch.float32
            and os.getenv('OPENDPD_DISABLE_ARENA_COMPILE', '0') != '1'):
        if _COMPILED is None:
            from models import CoreModel
            # Compile the unbound function: deepcopy never captures another
            # model's weights, and the three seeds can share compiled kernels.
            _COMPILED = torch.compile(CoreModel.forward, fullgraph=True,
                                       mode='max-autotune-no-cudagraphs')
        return _COMPILED(self, x, h_0)
    return self._arena_eager_forward(x, h_0)


def enable(model):
    # In particular Blackwell cuDNN can otherwise select TF32 for these small
    # recurrent matrices. Arena's FP32 contract applies to both train and test.
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    if model.backbone_type in ('deltagru', 'deltajanet', 'tres_deltagru'):
        layer = model.backbone.rnn
        if layer.num_layers == 1 and layer.th_x == layer.th_h == 0 and not layer.debug:
            layer._arena_eager_forward = layer.forward
            layer.forward = MethodType(_dense_delta_forward, layer)
    elif model.backbone_type=='bojanet':
        model.backbone._arena_eager_forward=model.backbone.forward
        model.backbone.forward=MethodType(_dense_bojanet_forward,model.backbone)
    elif model.backbone_type in COMPILED_KEYS:
        model._arena_eager_forward = model.forward
        model.forward = MethodType(_compiled_forward, model)
    return model
