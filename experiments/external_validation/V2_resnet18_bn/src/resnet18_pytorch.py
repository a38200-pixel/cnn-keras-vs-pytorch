"""Direct PyTorch implementation of the V2 semantic ResNet18."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from common.controlled_batchnorm import COMMON_BN_EPSILON, COMMON_BN_UPDATE_RATE
from v2_config import NUM_CLASSES


def build_torch_resnet18():
    import torch
    from torch import nn

    class CommonBatchNorm(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float32))
            self.register_buffer("running_mean", torch.zeros(channels, dtype=torch.float32))
            self.register_buffer("running_var", torch.ones(channels, dtype=torch.float32))

        def forward(self, inputs):
            if self.training:
                mean = torch.mean(inputs, dim=(0, 2, 3))
                centered = inputs - mean[None, :, None, None]
                variance = torch.mean(torch.square(centered), dim=(0, 2, 3))
                with torch.no_grad():
                    self.running_mean.mul_(1.0 - COMMON_BN_UPDATE_RATE).add_(
                        mean, alpha=COMMON_BN_UPDATE_RATE,
                    )
                    self.running_var.mul_(1.0 - COMMON_BN_UPDATE_RATE).add_(
                        variance, alpha=COMMON_BN_UPDATE_RATE,
                    )
            else:
                mean = self.running_mean
                variance = self.running_var
                centered = inputs - mean[None, :, None, None]
            x_hat = centered * torch.rsqrt(variance + COMMON_BN_EPSILON)[None, :, None, None]
            output = x_hat * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            self.last_trace = {
                "input": inputs, "batch_mean": mean, "batch_variance": variance,
                "x_hat": x_hat, "output": output,
            }
            return output

    class BasicBlock(nn.Module):
        def __init__(self, in_channels: int, channels: int, stride: int, semantic_prefix: str):
            super().__init__()
            self.semantic_prefix = semantic_prefix
            self.pad1 = nn.ZeroPad2d(1)
            self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=0, bias=False)
            self.bn1 = CommonBatchNorm(channels)
            self.relu1 = nn.ReLU()
            self.pad2 = nn.ZeroPad2d(1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=0, bias=False)
            self.bn2 = CommonBatchNorm(channels)
            self.has_projection = stride != 1 or in_channels != channels
            if self.has_projection:
                self.shortcut_conv = nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False)
                self.shortcut_bn = CommonBatchNorm(channels)
            self.relu_out = nn.ReLU()

        def forward(self, inputs, trace, record):
            prefix = self.semantic_prefix
            record(trace, f"{prefix}.input", inputs)
            x = self.conv1(self.pad1(inputs))
            record(trace, f"{prefix}.conv1", x)
            x = self.bn1(x)
            for field, value in self.bn1.last_trace.items():
                record(trace, f"{prefix}.bn1.{field}", value)
            x = self.relu1(x)
            record(trace, f"{prefix}.relu1", x)
            x = self.conv2(self.pad2(x))
            record(trace, f"{prefix}.conv2", x)
            x = self.bn2(x)
            for field, value in self.bn2.last_trace.items():
                record(trace, f"{prefix}.bn2.{field}", value)
            record(trace, f"{prefix}.main", x)
            if self.has_projection:
                shortcut = self.shortcut_conv(inputs)
                record(trace, f"{prefix}.shortcut.conv", shortcut)
                shortcut = self.shortcut_bn(shortcut)
                for field, value in self.shortcut_bn.last_trace.items():
                    record(trace, f"{prefix}.shortcut.bn.{field}", value)
                record(trace, f"{prefix}.shortcut.output", shortcut)
            else:
                shortcut = inputs
                record(trace, f"{prefix}.shortcut.identity", shortcut)
            added = x + shortcut
            record(trace, f"{prefix}.add", added)
            output = self.relu_out(added)
            record(trace, f"{prefix}.relu_out", output)
            return output

        def state_bindings(self):
            prefix = self.semantic_prefix
            result = {
                f"{prefix}.conv1.kernel": self.conv1.weight,
                f"{prefix}.bn1.gamma": self.bn1.weight,
                f"{prefix}.bn1.beta": self.bn1.bias,
                f"{prefix}.bn1.mean": self.bn1.running_mean,
                f"{prefix}.bn1.variance": self.bn1.running_var,
                f"{prefix}.conv2.kernel": self.conv2.weight,
                f"{prefix}.bn2.gamma": self.bn2.weight,
                f"{prefix}.bn2.beta": self.bn2.bias,
                f"{prefix}.bn2.mean": self.bn2.running_mean,
                f"{prefix}.bn2.variance": self.bn2.running_var,
            }
            if self.has_projection:
                result.update({
                    f"{prefix}.shortcut.conv.kernel": self.shortcut_conv.weight,
                    f"{prefix}.shortcut.bn.gamma": self.shortcut_bn.weight,
                    f"{prefix}.shortcut.bn.beta": self.shortcut_bn.bias,
                    f"{prefix}.shortcut.bn.mean": self.shortcut_bn.running_mean,
                    f"{prefix}.shortcut.bn.variance": self.shortcut_bn.running_var,
                })
            return result

    class SemanticResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem_pad = nn.ZeroPad2d(3)
            self.stem_conv = nn.Conv2d(3, 64, 7, stride=2, padding=0, bias=False)
            self.stem_bn = CommonBatchNorm(64)
            self.stem_relu = nn.ReLU()
            self.stem_pool_pad = nn.ZeroPad2d(1)
            self.stem_maxpool = nn.MaxPool2d(3, stride=2, padding=0)
            self.blocks = nn.ModuleList()
            in_channels = 64
            for stage_index, channels in enumerate((64, 128, 256, 512), 1):
                for block_index in (1, 2):
                    stride = 2 if stage_index > 1 and block_index == 1 else 1
                    block = BasicBlock(
                        in_channels, channels, stride,
                        semantic_prefix=f"stage{stage_index}.block{block_index}",
                    )
                    self.blocks.append(block)
                    in_channels = channels
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(512, NUM_CLASSES)
            self.trace_gradients = False
            self.gradient_trace_names = set()
            self.last_trace = {}

        def _record(self, trace, name, value):
            if self.trace_gradients and name in self.gradient_trace_names and value.requires_grad:
                value.retain_grad()
            trace[name] = value

        def forward(self, inputs):
            trace = {"input": inputs}
            x = self.stem_conv(self.stem_pad(inputs))
            self._record(trace, "stem.conv", x)
            x = self.stem_bn(x)
            for field, value in self.stem_bn.last_trace.items():
                self._record(trace, f"stem.bn.{field}", value)
            x = self.stem_relu(x)
            self._record(trace, "stem.relu", x)
            x = self.stem_maxpool(self.stem_pool_pad(x))
            self._record(trace, "stem.maxpool", x)
            for block in self.blocks:
                x = block(x, trace, self._record)
            x = self.gap(x).flatten(1)
            self._record(trace, "gap", x)
            logits = self.classifier(x)
            self._record(trace, "classifier.logits", logits)
            self.last_trace = trace
            return logits

        def state_bindings(self):
            result = {
                "stem.conv.kernel": self.stem_conv.weight,
                "stem.bn.gamma": self.stem_bn.weight,
                "stem.bn.beta": self.stem_bn.bias,
                "stem.bn.mean": self.stem_bn.running_mean,
                "stem.bn.variance": self.stem_bn.running_var,
            }
            for block in self.blocks:
                result.update(block.state_bindings())
            result["classifier.kernel"] = self.classifier.weight
            result["classifier.bias"] = self.classifier.bias
            return result

    return SemanticResNet18()
