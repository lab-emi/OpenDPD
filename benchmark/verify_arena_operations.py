"""Check the analytic Arena operation count against an instrumented forward pass.

arena_ops derives every count from a model's equations. This script runs the
real PyTorch module on a short sequence under a dispatch-level counter and
compares the products executed inside matrix and convolution kernels with the
ledger's ``kernel_mul`` for every registered sweep configuration. It also
compares the analytic parameter count with the instantiated module. No training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from opendpd.core import arena, arena_ops
from opendpd.core.arena_engine import build_model, parameter_count

SAMPLES = 40


class KernelProducts(TorchDispatchMode):
    """Real multiplications performed by mm/addmm/bmm and convolution kernels."""

    def __init__(self):
        super().__init__()
        self.products = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        name = func._overloadpacket.__name__
        if name == "mm":
            self.products += args[0].shape[0] * args[0].shape[1] * args[1].shape[1]
        elif name == "addmm":
            self.products += args[1].shape[0] * args[1].shape[1] * args[2].shape[1]
        elif name in ("bmm", "baddbmm"):
            left, right = args[-2], args[-1]
            self.products += left.shape[0] * left.shape[1] * left.shape[2] * right.shape[2]
        elif "convolution" in name and "backward" not in name:
            weight = args[1]
            self.products += output.numel() * (weight.numel() // weight.shape[0])
        elif name in ("gru", "lstm", "mkldnn_rnn_layer", "_thnn_fused_gru_cell", "_thnn_fused_lstm_cell"):
            raise RuntimeError(f"Opaque recurrent kernel {name}: disable fused RNN kernels for this check")
        return output


def verify(key, budget):
    parameters = arena.model_parameters(key, budget)
    if parameters is None:
        return None
    base = arena._base(key)
    cost = arena_ops.count(base, parameters)
    torch.manual_seed(0)
    model = build_model(base, parameters).eval()
    counted = None
    if base not in arena.DETERMINISTIC:
        with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False), KernelProducts() as counter:
            model(torch.randn(1, SAMPLES, 2) * 0.3)
        counted = counter.products
    return dict(backbone=key, budget=budget, parameters=arena_ops.parameter_count(base, parameters),
                instantiated_parameters=parameter_count(model), mul=cost["mul"], add=cost["add"], ops=cost["ops"],
                kernel_mul_per_sample=cost["kernel_mul"],
                instrumented_kernel_mul_per_sample=None if counted is None else counted / SAMPLES)


def report():
    rows = [row for model in arena.bundled_backbones() for budget in arena.BUDGETS
            if (row := verify(model.key, budget)) is not None]
    problems = [f"{row['backbone']}@{row['budget']}: parameters {row['parameters']} != {row['instantiated_parameters']}"
                for row in rows if row["parameters"] != row["instantiated_parameters"]]
    problems += [f"{row['backbone']}@{row['budget']}: kernel products {row['kernel_mul_per_sample']} != "
                 f"{row['instrumented_kernel_mul_per_sample']}" for row in rows
                 if row["instrumented_kernel_mul_per_sample"] not in (None, row["kernel_mul_per_sample"])]
    return dict(samples=SAMPLES, configurations=len(rows), instrumented=sum(
        row["instrumented_kernel_mul_per_sample"] is not None for row in rows), problems=problems, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = report()
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return int(bool(result["problems"]))


if __name__ == "__main__":
    raise SystemExit(main())
