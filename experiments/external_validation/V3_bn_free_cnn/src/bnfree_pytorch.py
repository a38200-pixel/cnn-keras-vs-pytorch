"""Direct PyTorch implementation of the V3 BN-free custom CNN."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")


def build_torch_bnfree():
    import torch
    from torch import nn

    class BNFreeCNN(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (3, 32, 64, 128, 256)
            for index in range(1, 5):
                setattr(self, f"conv{index}", nn.Conv2d(
                    channels[index - 1], channels[index], 3,
                    stride=1, padding=1, bias=False,
                ))
                setattr(self, f"relu{index}", nn.ReLU())
                setattr(self, f"pool{index}", nn.MaxPool2d(2))
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc128 = nn.Linear(256, 128)
            self.fc128_relu = nn.ReLU()
            self.logits_layer = nn.Linear(128, 8)
            self.trace_gradients = False
            self.gradient_trace_names = set()
            self.last_trace = {}

        def _record(self, trace, name, value):
            if self.trace_gradients and name in self.gradient_trace_names and value.requires_grad:
                value.retain_grad()
            trace[name] = value

        def forward(self, inputs):
            trace = {"input": inputs}
            x = inputs
            for index in range(1, 5):
                x = getattr(self, f"conv{index}")(x)
                self._record(trace, f"conv{index}", x)
                x = getattr(self, f"relu{index}")(x)
                self._record(trace, f"relu{index}", x)
                x = getattr(self, f"pool{index}")(x)
                self._record(trace, f"pool{index}", x)
            x = self.gap(x).flatten(1)
            self._record(trace, "gap", x)
            x = self.fc128(x)
            self._record(trace, "fc128.preactivation", x)
            x = self.fc128_relu(x)
            self._record(trace, "fc128.relu", x)
            logits = self.logits_layer(x)
            self._record(trace, "logits", logits)
            self.last_trace = trace
            return logits

        def state_bindings(self):
            result = {
                f"conv{index}/kernel": getattr(self, f"conv{index}").weight
                for index in range(1, 5)
            }
            result.update({
                "fc128/kernel": self.fc128.weight,
                "fc128/bias": self.fc128.bias,
                "logits/kernel": self.logits_layer.weight,
                "logits/bias": self.logits_layer.bias,
            })
            return result

    return BNFreeCNN()
