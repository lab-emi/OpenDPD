"""Replay small, stateless CUDA networks without changing optimizer arithmetic.

Capture contains the existing forward, loss, backward and gradient clipping.
Adam/AdamW/SGD and the scheduler still execute normally, including LR changes.
Only reviewed native backbones qualify; all other workloads keep eager mode.
"""
from __future__ import annotations

import os

import torch
from torch import nn


def supported_model(net):
    from models import CoreModel, CascadedModel
    supported = {"gru", "dgru", "lstm", "tres_gru", "rvtdcnn"}
    if type(net) is CoreModel:
        return net.backbone_type in supported
    if type(net) is CascadedModel:
        return (supported_model(net.dpd_model) and supported_model(net.pa_model)
                and not any(p.requires_grad for p in net.pa_model.parameters()))
    return False


def make_fast_step(net, criterion, parameters, grad_clip):
    if (os.getenv("OPENDPD_DISABLE_CUDA_FAST_PATH", "0") == "1"
            or not supported_model(net)
            or type(criterion) not in (nn.MSELoss, nn.L1Loss)
            or criterion.reduction != "mean"
            or torch.are_deterministic_algorithms_enabled()
            or not parameters
            or tuple(map(id, parameters)) != tuple(id(p) for p in net.parameters() if p.requires_grad)
            or any(not p.is_cuda or p.dtype != torch.float32 for p in net.parameters())
            or any((isinstance(m, nn.Dropout) and m.p > 0)
                   or (isinstance(m, nn.RNNBase) and m.dropout > 0) for m in net.modules())
            or any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
                   or m._backward_pre_hooks for m in net.modules())):
        return None
    return FastCudaStep(net, criterion, parameters, grad_clip)


class FastCudaStep:
    def __init__(self, net, criterion, parameters, grad_clip):
        self.net, self.criterion = net, criterion
        self.parameters, self.grad_clip = parameters, grad_clip
        self.graph = None
        self.failed = False
        self.shape = None

    def body(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()
        loss = self.criterion(self.net(self.x), self.y)
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
        return loss

    def capture(self, x, y):
        # No optimizer steps occur during warmup or capture. RNG state is
        # restored, and the next replay clears all warmup gradients.
        from backbones.cuda_graph_frozen_dgru import disable_cuda_graph_frozen_dgru
        with (torch.cuda.device(x.device), torch.random.fork_rng(devices=[x.device.index]),
              disable_cuda_graph_frozen_dgru()):
            for module in self.net.modules():
                if isinstance(module, nn.RNNBase):
                    module.flatten_parameters()
            self.x, self.y = x.detach().clone(), y.detach().clone()
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.body()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize(x.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.loss = self.body().detach()
            self.graph = graph
            self.shape = (tuple(x.shape), tuple(y.shape))
            self.pointers = tuple((p.data_ptr(), p.grad.data_ptr()) for p in self.parameters)

    def __call__(self, x, y):
        if (self.failed or not self.net.training or not torch.is_grad_enabled()
                or not x.is_cuda or not y.is_cuda or x.device != y.device
                or x.dtype != torch.float32 or y.dtype != torch.float32
                or x.requires_grad or y.requires_grad or torch.is_autocast_enabled()):
            return None
        shape = (tuple(x.shape), tuple(y.shape))
        # A short final batch stays eager; it must not change the captured
        # gradient buffers, so net_train clears them in place for this path.
        if self.shape is not None and shape != self.shape:
            return None
        if self.graph is not None and self.pointers != tuple(
                (p.data_ptr(), p.grad.data_ptr() if p.grad is not None else None) for p in self.parameters):
            self.graph = None
        if self.graph is None:
            try:
                self.capture(x, y)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                self.graph = None
                self.failed = True
                print(f"[OpenDPD] CUDA replay unavailable; using eager training ({type(error).__name__}).", flush=True)
                return None
            print("[OpenDPD] CUDA replay enabled for forward, backward and clipping; optimizer unchanged.", flush=True)
        self.x.copy_(x, non_blocking=True)
        self.y.copy_(y, non_blocking=True)
        self.graph.replay()
        # Each batch needs its own scalar: the graph overwrites self.loss.
        return self.loss.detach().clone()
