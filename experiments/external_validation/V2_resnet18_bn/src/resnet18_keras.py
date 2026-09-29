"""Direct Keras implementation of the V2 semantic ResNet18."""

from __future__ import annotations

from common.controlled_batchnorm import COMMON_BN_EPSILON, COMMON_BN_UPDATE_RATE
from v2_config import NUM_CLASSES


def build_keras_resnet18():
    import tensorflow as tf
    from tensorflow.keras import layers

    class CommonBatchNorm(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.epsilon = tf.constant(COMMON_BN_EPSILON, tf.float32)
            self.update_rate = tf.constant(COMMON_BN_UPDATE_RATE, tf.float32)

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

        def call(self, inputs, training=False):
            if training:
                mean = tf.reduce_mean(inputs, axis=(0, 1, 2))
                centered = tf.subtract(inputs, mean)
                variance = tf.reduce_mean(tf.square(centered), axis=(0, 1, 2))
                keep = tf.constant(1.0, tf.float32) - self.update_rate
                self.moving_mean.assign(keep * self.moving_mean + self.update_rate * mean)
                self.moving_variance.assign(keep * self.moving_variance + self.update_rate * variance)
            else:
                mean = self.moving_mean
                variance = self.moving_variance
                centered = tf.subtract(inputs, mean)
            x_hat = centered * tf.math.rsqrt(variance + self.epsilon)
            output = x_hat * self.gamma + self.beta
            self.last_trace = {
                "input": inputs, "batch_mean": mean, "batch_variance": variance,
                "x_hat": x_hat, "output": output,
            }
            return output

    class BasicBlock(layers.Layer):
        def __init__(self, in_channels: int, channels: int, stride: int, semantic_prefix: str, **kwargs):
            super().__init__(**kwargs)
            self.semantic_prefix = semantic_prefix
            self.pad1 = layers.ZeroPadding2D(1, name="pad1")
            self.conv1 = layers.Conv2D(
                channels, 3, strides=stride, padding="valid", use_bias=False, name="conv1",
            )
            self.bn1 = CommonBatchNorm(name="bn1")
            self.relu1 = layers.ReLU(name="relu1")
            self.pad2 = layers.ZeroPadding2D(1, name="pad2")
            self.conv2 = layers.Conv2D(channels, 3, padding="valid", use_bias=False, name="conv2")
            self.bn2 = CommonBatchNorm(name="bn2")
            self.has_projection = stride != 1 or in_channels != channels
            if self.has_projection:
                self.shortcut_conv = layers.Conv2D(
                    channels, 1, strides=stride, padding="valid", use_bias=False,
                    name="shortcut_conv",
                )
                self.shortcut_bn = CommonBatchNorm(name="shortcut_bn")
            self.relu_out = layers.ReLU(name="relu_out")

        def call(self, inputs, training=False):
            prefix = self.semantic_prefix
            trace = {}
            trace[f"{prefix}.input"] = inputs
            x = self.conv1(self.pad1(inputs))
            trace[f"{prefix}.conv1"] = x
            x = self.bn1(x, training=training)
            for field, value in self.bn1.last_trace.items():
                trace[f"{prefix}.bn1.{field}"] = value
            x = self.relu1(x)
            trace[f"{prefix}.relu1"] = x
            x = self.conv2(self.pad2(x))
            trace[f"{prefix}.conv2"] = x
            x = self.bn2(x, training=training)
            for field, value in self.bn2.last_trace.items():
                trace[f"{prefix}.bn2.{field}"] = value
            trace[f"{prefix}.main"] = x
            if self.has_projection:
                shortcut = self.shortcut_conv(inputs)
                trace[f"{prefix}.shortcut.conv"] = shortcut
                shortcut = self.shortcut_bn(shortcut, training=training)
                for field, value in self.shortcut_bn.last_trace.items():
                    trace[f"{prefix}.shortcut.bn.{field}"] = value
                trace[f"{prefix}.shortcut.output"] = shortcut
            else:
                shortcut = inputs
                trace[f"{prefix}.shortcut.identity"] = shortcut
            added = x + shortcut
            trace[f"{prefix}.add"] = added
            output = self.relu_out(added)
            trace[f"{prefix}.relu_out"] = output
            return output, trace

        def state_bindings(self):
            prefix = self.semantic_prefix
            result = {
                f"{prefix}.conv1.kernel": self.conv1.kernel,
                f"{prefix}.bn1.gamma": self.bn1.gamma,
                f"{prefix}.bn1.beta": self.bn1.beta,
                f"{prefix}.bn1.mean": self.bn1.moving_mean,
                f"{prefix}.bn1.variance": self.bn1.moving_variance,
                f"{prefix}.conv2.kernel": self.conv2.kernel,
                f"{prefix}.bn2.gamma": self.bn2.gamma,
                f"{prefix}.bn2.beta": self.bn2.beta,
                f"{prefix}.bn2.mean": self.bn2.moving_mean,
                f"{prefix}.bn2.variance": self.bn2.moving_variance,
            }
            if self.has_projection:
                result.update({
                    f"{prefix}.shortcut.conv.kernel": self.shortcut_conv.kernel,
                    f"{prefix}.shortcut.bn.gamma": self.shortcut_bn.gamma,
                    f"{prefix}.shortcut.bn.beta": self.shortcut_bn.beta,
                    f"{prefix}.shortcut.bn.mean": self.shortcut_bn.moving_mean,
                    f"{prefix}.shortcut.bn.variance": self.shortcut_bn.moving_variance,
                })
            return result

    class SemanticResNet18(tf.keras.Model):
        def __init__(self):
            super().__init__(name="v2_semantic_resnet18")
            self.stem_pad = layers.ZeroPadding2D(3, name="stem_pad")
            self.stem_conv = layers.Conv2D(64, 7, strides=2, padding="valid", use_bias=False, name="stem_conv")
            self.stem_bn = CommonBatchNorm(name="stem_bn")
            self.stem_relu = layers.ReLU(name="stem_relu")
            # Post-ReLU values are non-negative, so explicit zero padding is
            # equivalent to the reference MaxPool padding=1 boundary behavior.
            self.stem_pool_pad = layers.ZeroPadding2D(1, name="stem_pool_pad")
            self.stem_maxpool = layers.MaxPooling2D(3, strides=2, padding="valid", name="stem_maxpool")
            self.blocks = []
            in_channels = 64
            for stage_index, channels in enumerate((64, 128, 256, 512), 1):
                for block_index in (1, 2):
                    stride = 2 if stage_index > 1 and block_index == 1 else 1
                    block = BasicBlock(
                        in_channels, channels, stride,
                        semantic_prefix=f"stage{stage_index}.block{block_index}",
                        name=f"stage{stage_index}_block{block_index}",
                    )
                    self.blocks.append(block)
                    in_channels = channels
            self.gap = layers.GlobalAveragePooling2D(name="gap")
            self.classifier = layers.Dense(NUM_CLASSES, name="classifier")
            self.last_trace = {}

        def call(self, inputs, training=False):
            trace = {"input": inputs}
            x = self.stem_conv(self.stem_pad(inputs))
            trace["stem.conv"] = x
            x = self.stem_bn(x, training=training)
            for field, value in self.stem_bn.last_trace.items():
                trace[f"stem.bn.{field}"] = value
            x = self.stem_relu(x)
            trace["stem.relu"] = x
            x = self.stem_maxpool(self.stem_pool_pad(x))
            trace["stem.maxpool"] = x
            for block in self.blocks:
                x, block_trace = block(x, training=training)
                trace.update(block_trace)
            x = self.gap(x)
            trace["gap"] = x
            logits = self.classifier(x)
            trace["classifier.logits"] = logits
            self.last_trace = trace
            return logits

        def state_bindings(self):
            result = {
                "stem.conv.kernel": self.stem_conv.kernel,
                "stem.bn.gamma": self.stem_bn.gamma,
                "stem.bn.beta": self.stem_bn.beta,
                "stem.bn.mean": self.stem_bn.moving_mean,
                "stem.bn.variance": self.stem_bn.moving_variance,
            }
            for block in self.blocks:
                result.update(block.state_bindings())
            result["classifier.kernel"] = self.classifier.kernel
            result["classifier.bias"] = self.classifier.bias
            return result

    model = SemanticResNet18()
    model(tf.zeros((1, 128, 128, 3), dtype=tf.float32), training=False)
    return model
