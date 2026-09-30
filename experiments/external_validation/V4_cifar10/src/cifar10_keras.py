"""Direct Keras V4 CIFAR-10 model using the existing CommonBN implementation."""

from __future__ import annotations

from common.controlled_batchnorm import build_keras_common_bn_model


def build_keras_cifar10():
    import tensorflow as tf
    from tensorflow.keras import layers

    CommonBN = type(build_keras_common_bn_model().get_layer("bn1"))

    class CIFAR10CNN(tf.keras.Model):
        def __init__(self):
            super().__init__(name="v4_cifar10_common_bn_cnn")
            self.convs, self.bns, self.relus, self.pools = [], [], [], []
            for index, channels in enumerate((32, 64, 128, 256), 1):
                conv = layers.Conv2D(channels, 3, padding="same", use_bias=False, name=f"conv{index}")
                bn = CommonBN(name=f"bn{index}")
                relu = layers.ReLU(name=f"relu{index}")
                pool = layers.MaxPooling2D(2, name=f"pool{index}")
                setattr(self, f"conv{index}", conv)
                setattr(self, f"bn{index}", bn)
                setattr(self, f"relu{index}", relu)
                setattr(self, f"pool{index}", pool)
                self.convs.append(conv); self.bns.append(bn); self.relus.append(relu); self.pools.append(pool)
            self.gap = layers.GlobalAveragePooling2D(name="gap")
            self.fc128 = layers.Dense(128, name="fc128")
            self.fc128_relu = layers.ReLU(name="fc128_relu")
            self.classifier = layers.Dense(10, name="classifier")
            self.last_trace = {}

        def call(self, inputs, training=False):
            trace = {"input": inputs}
            x = inputs
            for index, (conv, bn, relu, pool) in enumerate(zip(self.convs, self.bns, self.relus, self.pools), 1):
                x = conv(x); trace[f"conv{index}"] = x
                x = bn(x, training=training)
                for field, value in bn.last_trace.items():
                    trace[f"bn{index}.{field}"] = value
                x = relu(x); trace[f"relu{index}"] = x
                x = pool(x); trace[f"pool{index}"] = x
            x = self.gap(x); trace["gap"] = x
            x = self.fc128(x); trace["fc128.preactivation"] = x
            x = self.fc128_relu(x); trace["fc128.relu"] = x
            logits = self.classifier(x); trace["logits"] = logits
            self.last_trace = trace
            return logits

        def state_bindings(self):
            result = {}
            for index, (conv, bn) in enumerate(zip(self.convs, self.bns), 1):
                result[f"conv{index}/kernel"] = conv.kernel
                result[f"bn{index}/gamma"] = bn.gamma
                result[f"bn{index}/beta"] = bn.beta
                result[f"bn{index}/mean"] = bn.moving_mean
                result[f"bn{index}/variance"] = bn.moving_variance
            result.update({
                "fc128/kernel": self.fc128.kernel, "fc128/bias": self.fc128.bias,
                "classifier/kernel": self.classifier.kernel, "classifier/bias": self.classifier.bias,
            })
            return result

    model = CIFAR10CNN()
    model(tf.zeros((1, 32, 32, 3), tf.float32), training=False)
    return model
