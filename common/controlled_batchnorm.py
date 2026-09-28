"""Common BatchNorm semantics implemented with each framework's tensor ops."""

from __future__ import annotations

from common.controlled_config import IMAGE_SIZE, NUM_CLASSES

COMMON_BN_EPSILON = 1e-3
COMMON_BN_UPDATE_RATE = 0.01
COMMON_BN_SEMANTICS_ID = "common_bn_population_variance_rate_0.01_v1"


def build_keras_common_bn_model():
    """Build the controlled CNN with explicit NHWC Common BatchNorm."""
    import tensorflow as tf
    from tensorflow.keras import layers

    class KerasCommonBatchNorm(layers.Layer):
        semantics_id = COMMON_BN_SEMANTICS_ID
        reduction_axes = (0, 1, 2)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.epsilon = tf.constant(COMMON_BN_EPSILON, dtype=tf.float32)
            self.update_rate = tf.constant(COMMON_BN_UPDATE_RATE, dtype=tf.float32)

        def build(self, input_shape):
            channels = int(input_shape[-1])
            self.gamma = self.add_weight(name="gamma", shape=(channels,), initializer="ones", trainable=True)
            self.beta = self.add_weight(name="beta", shape=(channels,), initializer="zeros", trainable=True)
            self.moving_mean = self.add_weight(
                name="moving_mean", shape=(channels,), initializer="zeros", trainable=False,
            )
            self.moving_variance = self.add_weight(
                name="moving_variance", shape=(channels,), initializer="ones", trainable=False,
            )
            super().build(input_shape)

        def call(self, inputs, training=None):
            if training:
                mean = tf.reduce_mean(inputs, axis=self.reduction_axes)
                centered = tf.subtract(inputs, mean)
                variance = tf.reduce_mean(tf.square(centered), axis=self.reduction_axes)
                keep = tf.constant(1.0, tf.float32) - self.update_rate
                self.moving_mean.assign(
                    tf.add(tf.multiply(keep, self.moving_mean), tf.multiply(self.update_rate, mean))
                )
                self.moving_variance.assign(
                    tf.add(tf.multiply(keep, self.moving_variance), tf.multiply(self.update_rate, variance))
                )
            else:
                mean = self.moving_mean
                variance = self.moving_variance
                centered = tf.subtract(inputs, mean)
            inverse_std = tf.math.rsqrt(tf.add(variance, self.epsilon))
            normalized = tf.multiply(centered, inverse_std)
            return tf.add(tf.multiply(normalized, self.gamma), self.beta)

    inputs = layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype="float32", name="input")
    x = inputs
    for index, channels in enumerate((32, 64, 128, 256), 1):
        x = layers.Conv2D(channels, 3, padding="same", use_bias=False, name=f"conv{index}")(x)
        x = KerasCommonBatchNorm(name=f"bn{index}")(x)
        x = layers.ReLU(name=f"relu{index}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{index}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, name="fc128")(x)
    x = layers.ReLU(name="dense128_relu")(x)
    outputs = layers.Dense(NUM_CLASSES, name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="controlled_common_bn_rgb_cnn")


def build_torch_common_bn_model():
    """Build the controlled CNN with explicit NCHW Common BatchNorm."""
    import torch
    from torch import nn

    class TorchCommonBatchNorm(nn.Module):
        semantics_id = COMMON_BN_SEMANTICS_ID
        reduction_axes = (0, 2, 3)

        def __init__(self, channels: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float32))
            self.register_buffer("running_mean", torch.zeros(channels, dtype=torch.float32))
            self.register_buffer("running_var", torch.ones(channels, dtype=torch.float32))
            self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.int64))
            self.epsilon = COMMON_BN_EPSILON
            self.update_rate = COMMON_BN_UPDATE_RATE

        def forward(self, inputs):
            if self.training:
                mean = torch.mean(inputs, dim=self.reduction_axes)
                centered = torch.sub(inputs, mean[None, :, None, None])
                variance = torch.mean(torch.square(centered), dim=self.reduction_axes)
                with torch.no_grad():
                    self.running_mean.mul_(1.0 - self.update_rate).add_(mean, alpha=self.update_rate)
                    self.running_var.mul_(1.0 - self.update_rate).add_(variance, alpha=self.update_rate)
                    self.num_batches_tracked.add_(1)
            else:
                mean = self.running_mean
                variance = self.running_var
                centered = torch.sub(inputs, mean[None, :, None, None])
            inverse_std = torch.rsqrt(torch.add(variance, self.epsilon))
            normalized = torch.mul(centered, inverse_std[None, :, None, None])
            return torch.add(
                torch.mul(normalized, self.weight[None, :, None, None]),
                self.bias[None, :, None, None],
            )

    class ControlledCommonBNCNN(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (3, 32, 64, 128, 256)
            for index in range(1, 5):
                setattr(self, f"conv{index}", nn.Conv2d(channels[index - 1], channels[index], 3, padding=1, bias=False))
                setattr(self, f"bn{index}", TorchCommonBatchNorm(channels[index]))
                setattr(self, f"relu{index}", nn.ReLU())
                setattr(self, f"pool{index}", nn.MaxPool2d(2))
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc128 = nn.Linear(256, 128)
            self.dense128_relu = nn.ReLU()
            self.logits = nn.Linear(128, NUM_CLASSES)

        def forward(self, inputs):
            x = inputs
            for index in range(1, 5):
                x = getattr(self, f"pool{index}")(
                    getattr(self, f"relu{index}")(
                        getattr(self, f"bn{index}")(
                            getattr(self, f"conv{index}")(x)
                        )
                    )
                )
            x = self.gap(x).flatten(1)
            return self.logits(self.dense128_relu(self.fc128(x)))

    return ControlledCommonBNCNN()


def keras_uses_native_batchnorm(model) -> bool:
    import tensorflow as tf

    return any(isinstance(layer, tf.keras.layers.BatchNormalization) for layer in model.layers)


def torch_uses_native_batchnorm(model) -> bool:
    import torch

    return any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
