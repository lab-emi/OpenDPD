"""Trusted PyTorch interpreter for validated backbone template v1 graphs."""
import torch
from torch import nn
from torch.nn import functional as F

from opendpd.core.backbone_template import DEFAULT_DEFINITION, parse_definition, validate_definition


class TemplateNetwork(nn.Module):
    def __init__(self, definition=DEFAULT_DEFINITION):
        super().__init__()
        self.definition = parse_definition(definition)
        widths = validate_definition(self.definition)["features"]
        self.layers = nn.ModuleList()
        for node in self.definition["nodes"]:
            op, inc = node["op"], widths[node["inputs"][0]]
            out = widths[node["id"]]
            if op == "linear":
                layer = nn.Linear(inc, out, bias=node.get("bias", True))
            elif op in ("gru", "lstm"):
                layer = (nn.GRU if op == "gru" else nn.LSTM)(inc, out, num_layers=node.get("layers", 1), batch_first=True)
            elif op == "conv1d":
                layer = nn.Conv1d(inc, out, node["kernel_size"], dilation=node.get("dilation", 1), bias=node.get("bias", True))
            elif op == "layer_norm":
                layer = nn.LayerNorm(inc)
            elif op == "dropout":
                layer = nn.Dropout(node["p"])
            else:
                layer = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}.get(op, nn.Identity)()
            self.layers.append(layer)

    def forward(self, x, h_0=None):
        # Frame-local state matches Studio's offline_segmented contract.
        values = {"input": x}
        for node, layer in zip(self.definition["nodes"], self.layers):
            inputs = [values[name] for name in node["inputs"]]
            value, op = inputs[0], node["op"]
            if op in ("gru", "lstm"):
                value, _ = layer(value)
            elif op == "conv1d":
                left = (node["kernel_size"] - 1) * node.get("dilation", 1)
                value = layer(F.pad(value.transpose(1, 2), (left, 0))).transpose(1, 2)
            elif op == "add":
                for other in inputs[1:]:
                    value = value + other
            elif op == "concat":
                value = torch.cat(inputs, dim=-1)
            elif op == "iq_features":
                # Stable at zero, including backpropagation through a frozen PA.
                power = value.square().sum(dim=-1, keepdim=True)
                envelope = (power + 1e-12).sqrt()
                value = torch.cat((value, envelope, power, envelope * power, power.square()), dim=-1)
            else:
                value = layer(value)
            values[node["id"]] = value
        return values[self.definition["output"]]
