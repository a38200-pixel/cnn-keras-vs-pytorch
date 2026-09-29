"""Direct Keras implementation of the V3 BN-free custom CNN."""

from __future__ import annotations


def build_keras_bnfree():
    import tensorflow as tf
    from tensorflow.keras import layers

    class BNFreeCNN(tf.keras.Model):
        def __init__(self):
            super().__init__(name="v3_bnfree_custom_cnn")
            channels = (32, 64, 128, 256)
            self.convs = []
            self.relus = []
            self.pools = []
            for index, count in enumerate(channels, 1):
                conv = layers.Conv2D(
                    count, 3, strides=1, padding="same", use_bias=False,
                    name=f"conv{index}",
                )
                relu = layers.ReLU(name=f"relu{index}")
                pool = layers.MaxPooling2D(2, name=f"pool{index}")
                setattr(self, f"conv{index}", conv)
                setattr(self, f"relu{index}", relu)
                setattr(self, f"pool{index}", pool)
                self.convs.append(conv)
                self.relus.append(relu)
                self.pools.append(pool)
            self.gap = layers.GlobalAveragePooling2D(name="gap")
            self.fc128 = layers.Dense(128, name="fc128")
            self.fc128_relu = layers.ReLU(name="fc128_relu")
            self.logits_layer = layers.Dense(8, name="logits")
            self.last_trace = {}

        def call(self, inputs, training=False):
            del training
            trace = {"input": inputs}
            x = inputs
            for index, (conv, relu, pool) in enumerate(
                zip(self.convs, self.relus, self.pools), 1,
            ):
                x = conv(x)
                trace[f"conv{index}"] = x
                x = relu(x)
                trace[f"relu{index}"] = x
                x = pool(x)
                trace[f"pool{index}"] = x
            x = self.gap(x)
            trace["gap"] = x
            x = self.fc128(x)
            trace["fc128.preactivation"] = x
            x = self.fc128_relu(x)
            trace["fc128.relu"] = x
            logits = self.logits_layer(x)
            trace["logits"] = logits
            self.last_trace = trace
            return logits

        def state_bindings(self):
            result = {
                f"conv{index}/kernel": conv.kernel
                for index, conv in enumerate(self.convs, 1)
            }
            result.update({
                "fc128/kernel": self.fc128.kernel,
                "fc128/bias": self.fc128.bias,
                "logits/kernel": self.logits_layer.kernel,
                "logits/bias": self.logits_layer.bias,
            })
            return result

    model = BNFreeCNN()
    model(tf.zeros((1, 128, 128, 3), dtype=tf.float32), training=False)
    return model
