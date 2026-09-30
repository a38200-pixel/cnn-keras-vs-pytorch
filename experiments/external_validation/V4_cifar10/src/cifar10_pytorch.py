"""Direct PyTorch V4 CIFAR-10 model using the existing CommonBN implementation."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from common.controlled_batchnorm import build_torch_common_bn_model


def build_torch_cifar10():
    import torch
    from torch import nn

    CommonBN = type(build_torch_common_bn_model().bn1)

    class CIFAR10CNN(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (3, 32, 64, 128, 256)
            for index in range(1, 5):
                setattr(self, f"conv{index}", nn.Conv2d(channels[index - 1], channels[index], 3, padding=1, bias=False))
                setattr(self, f"bn{index}", CommonBN(channels[index]))
                setattr(self, f"relu{index}", nn.ReLU())
                setattr(self, f"pool{index}", nn.MaxPool2d(2))
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc128 = nn.Linear(256, 128)
            self.fc128_relu = nn.ReLU()
            self.classifier = nn.Linear(128, 10)
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
                x = getattr(self, f"conv{index}")(x); self._record(trace, f"conv{index}", x)
                bn = getattr(self, f"bn{index}"); x = bn(x)
                for field, value in bn.last_trace.items():
                    self._record(trace, f"bn{index}.{field}", value)
                x = getattr(self, f"relu{index}")(x); self._record(trace, f"relu{index}", x)
                x = getattr(self, f"pool{index}")(x); self._record(trace, f"pool{index}", x)
            x = self.gap(x).flatten(1); self._record(trace, "gap", x)
            x = self.fc128(x); self._record(trace, "fc128.preactivation", x)
            x = self.fc128_relu(x); self._record(trace, "fc128.relu", x)
            logits = self.classifier(x); self._record(trace, "logits", logits)
            self.last_trace = trace
            return logits

        def state_bindings(self):
            result = {}
            for index in range(1, 5):
                conv, bn = getattr(self, f"conv{index}"), getattr(self, f"bn{index}")
                result[f"conv{index}/kernel"] = conv.weight
                result[f"bn{index}/gamma"] = bn.weight
                result[f"bn{index}/beta"] = bn.bias
                result[f"bn{index}/mean"] = bn.running_mean
                result[f"bn{index}/variance"] = bn.running_var
            result.update({
                "fc128/kernel": self.fc128.weight, "fc128/bias": self.fc128.bias,
                "classifier/kernel": self.classifier.weight, "classifier/bias": self.classifier.bias,
            })
            return result

    return CIFAR10CNN()
