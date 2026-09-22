"""Perform exactly one aligned optimizer step, then compare every activation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from compare_initial import RESULTS, SEEDS, build_pair, compare_models, shared_batch
from common.environment_utils import select_torch_device


def main() -> None:
    import tensorflow as tf
    import torch
    RESULTS.mkdir(parents=True, exist_ok=True); batch, labels = shared_batch()
    for seed in SEEDS:
        keras_model, torch_model = build_pair(seed)
        logits_model = tf.keras.Model(keras_model.input, keras_model.get_layer("logits").output)
        keras_optimizer = tf.keras.optimizers.Adam(.001, beta_1=.9, beta_2=.999, epsilon=1e-7)
        with tf.GradientTape() as tape:
            logits = logits_model(batch, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
        keras_optimizer.apply_gradients(zip(tape.gradient(loss, keras_model.trainable_variables), keras_model.trainable_variables))
        torch_model.train(); optimizer = torch.optim.Adam(torch_model.parameters(), lr=.001, betas=(.9, .999), eps=1e-7)
        device = select_torch_device()
        inputs = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        targets = torch.from_numpy(labels).to(device)
        optimizer.zero_grad(); torch_loss = torch.nn.CrossEntropyLoss()(torch_model(inputs), targets)
        torch_loss.backward(); optimizer.step()
        rows = compare_models(keras_model, torch_model, batch, seed, "one_step")
        (RESULTS / f"one_step_seed{seed}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(seed, rows[-1])


if __name__ == "__main__": main()
