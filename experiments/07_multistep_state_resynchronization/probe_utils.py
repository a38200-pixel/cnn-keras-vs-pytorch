"""Shared read-only checkpoint and one-step probe machinery for Experiment 07."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from common.augmentation_runtime import augmentation_parameters
from common.common_adam_control import (
    TRAINABLE_NAMES, keras_gradients, load_keras_trainable, load_torch_trainable,
    torch_gradients, trainable_values,
)
from common.controlled_batchnorm import build_keras_common_bn_model, build_torch_common_bn_model
from common.controlled_checkpoint import verify_checkpoint
from common.controlled_data import canonical_image, canonical_rows, tensor_hash
from common.controlled_initialization import (
    export_keras_weights, export_torch_weights, load_keras_weights, load_torch_weights,
)
from common.environment_utils import select_torch_device
from common.state_resynchronization import (
    BN_BETA_NAMES, BN_GAMMA_NAMES, BN_MEAN_NAMES, BN_VARIANCE_NAMES,
    adam_from_checkpoint, adam_m_state, adam_v_state, compare_group, state_fingerprint,
    states_exact,
)
from common.trace_utils import assert_finite, compare, write_csv
from experiment_config import (
    BATCH_SIZE, CHECKPOINTS, CHECKPOINT_STEPS, PARENT_CONFIG_SHA256, PARENT_RESULTS,
    PROBE_AUGMENTATION_EPOCH, PROBE_AUGMENTATION_SEED, PROBE_ID, RESULTS, SEEDS,
    anchors, config_hash,
)

EXPECTED_MODEL_KEYS = {
    *(f"conv{index}/kernel" for index in range(1, 5)),
    *(f"bn{index}/{field}" for index in range(1, 5)
      for field in ("gamma", "beta", "mean", "variance")),
    "fc128/kernel", "fc128/bias", "logits/kernel", "logits/bias",
}


def _selection_key(sample_id: str) -> str:
    return hashlib.sha256(f"{PROBE_ID}\0{sample_id}".encode("utf-8")).hexdigest()


def fixed_probe_batch(write_manifest: bool = False):
    rows = canonical_rows("train")
    selected = sorted(rows, key=lambda row: (_selection_key(str(row["sample_id"])), str(row["sample_id"])))[:BATCH_SIZE]
    images = np.stack([
        canonical_image(row, PROBE_AUGMENTATION_SEED, PROBE_AUGMENTATION_EPOCH, True)
        for row in selected
    ]).astype(np.float32, copy=False)
    images = np.ascontiguousarray(images)
    labels = np.asarray([row["label"] for row in selected], dtype=np.int32)
    batch_hash = tensor_hash(images)
    manifest = []
    for index, (row, image) in enumerate(zip(selected, images)):
        flip, angle = augmentation_parameters(
            str(row["sample_id"]), PROBE_AUGMENTATION_SEED, PROBE_AUGMENTATION_EPOCH,
        )
        manifest.append({
            "probe_id": PROBE_ID,
            "batch_index": index,
            "sample_id": row["sample_id"],
            "label": row["label"],
            "tensor_hash": tensor_hash(image),
            "batch_tensor_hash": batch_hash,
            "flip": bool(flip),
            "rotation_degrees": float(angle),
            "augmentation_seed": PROBE_AUGMENTATION_SEED,
            "augmentation_epoch": PROBE_AUGMENTATION_EPOCH,
            "dtype": str(image.dtype),
            "shape": "128x128x3",
        })
    if write_manifest:
        write_csv(RESULTS / "probe_batch_manifest.csv", manifest)
    return images, labels, manifest


def checkpoint_index() -> dict[tuple[int, str, str], dict[str, str]]:
    path = PARENT_RESULTS / "checkpoints/checkpoint_manifest.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        (seed, framework, checkpoint)
        for seed in SEEDS for framework in ("keras", "pytorch") for checkpoint in CHECKPOINTS
    }
    indexed = {(int(row["seed"]), row["framework"], row["checkpoint_name"]): row for row in rows}
    if set(indexed) != expected or len(rows) != 42:
        raise RuntimeError("Experiment 06 checkpoint manifest must contain exactly 42 expected entries")
    if any(row["config_sha256"] != PARENT_CONFIG_SHA256 for row in rows):
        raise RuntimeError("Experiment 06 checkpoint config hash mismatch")
    return indexed


def load_checkpoint(seed: int, framework: str, checkpoint: str):
    row = checkpoint_index()[(seed, framework, checkpoint)]
    model_state, optimizer_state = verify_checkpoint(
        PARENT_RESULTS / row["model_file"],
        PARENT_RESULTS / row["optimizer_file"],
        PARENT_RESULTS / row["metadata_file"],
        expected_config_hash=PARENT_CONFIG_SHA256,
    )
    metadata = json.loads((PARENT_RESULTS / row["metadata_file"]).read_text(encoding="utf-8"))
    step = int(metadata["optimizer_step"])
    if step != CHECKPOINT_STEPS[checkpoint]:
        raise RuntimeError(f"Checkpoint step mismatch: {seed}/{framework}/{checkpoint}")
    if set(model_state) != EXPECTED_MODEL_KEYS:
        raise RuntimeError(f"Checkpoint model mapping mismatch: {seed}/{framework}/{checkpoint}")
    assert_finite(model_state, "Experiment 06 model checkpoint")
    assert_finite(optimizer_state, "Experiment 06 optimizer checkpoint")
    return model_state, optimizer_state, step, metadata


def require_preflight() -> dict[str, object]:
    path = RESULTS / "preflight/resync_precheck.json"
    if not path.exists():
        raise RuntimeError("Run resync_preflight.py first")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("resync_precheck") != "VALID" or not report.get("probe_ready"):
        raise RuntimeError(f"Experiment 07 preflight is not VALID: {report.get('errors')}")
    if report.get("config_sha256") != config_hash():
        raise RuntimeError("Experiment 07 config changed since preflight")
    return report


class ProbeRunner:
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        import tensorflow as tf
        import torch

        self.tf = tf
        self.torch = torch
        tf.keras.backend.set_floatx("float32")
        self.device = select_torch_device()
        if not tf.config.list_physical_devices("GPU") or self.device.type != "mps":
            raise RuntimeError("TensorFlow Metal GPU and PyTorch MPS are required")
        self.images = np.ascontiguousarray(images.astype(np.float32, copy=False))
        self.labels = np.asarray(labels, dtype=np.int32)
        self.kx = tf.convert_to_tensor(self.images, dtype=tf.float32)
        self.ky = tf.convert_to_tensor(self.labels, dtype=tf.int32)
        self.px = torch.from_numpy(self.images.transpose(0, 3, 1, 2).copy()).to(self.device)
        self.py = torch.from_numpy(self.labels.astype(np.int64)).to(self.device)
        restored = self.px.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.input_exact = bool(np.array_equal(self.images, restored))
        self.label_exact = bool(np.array_equal(self.labels, self.py.detach().cpu().numpy().astype(np.int32)))
        if not self.input_exact or not self.label_exact:
            raise RuntimeError("Fixed diagnostic input/label conversion mismatch")
        self.keras_model = build_keras_common_bn_model()
        self.torch_model = build_torch_common_bn_model().to(self.device)

    def validate_resync(
        self, model_state: dict[str, np.ndarray], optimizer_state: dict[str, np.ndarray], step: int,
    ) -> dict[str, object]:
        load_keras_weights(self.keras_model, model_state)
        load_torch_weights(self.torch_model, model_state)
        keras_state = export_keras_weights(self.keras_model)
        torch_state = export_torch_weights(self.torch_model)
        keras_adam = adam_from_checkpoint(model_state, optimizer_state, step)
        torch_adam = adam_from_checkpoint(model_state, optimizer_state, step)
        checks = {
            "model_weight_exact": states_exact(keras_state, torch_state, TRAINABLE_NAMES),
            "adam_m_exact": states_exact(keras_adam.m, torch_adam.m, TRAINABLE_NAMES),
            "adam_v_exact": states_exact(keras_adam.v, torch_adam.v, TRAINABLE_NAMES),
            "adam_step_exact": keras_adam.step == torch_adam.step == step,
            "bn_gamma_exact": states_exact(keras_state, torch_state, BN_GAMMA_NAMES),
            "bn_beta_exact": states_exact(keras_state, torch_state, BN_BETA_NAMES),
            "bn_running_mean_exact": states_exact(keras_state, torch_state, BN_MEAN_NAMES),
            "bn_running_variance_exact": states_exact(keras_state, torch_state, BN_VARIANCE_NAMES),
            "input_exact": self.input_exact,
            "label_exact": self.label_exact,
        }
        checks["all_exact"] = all(checks.values())
        if not checks["all_exact"]:
            raise RuntimeError(f"Full-state re-synchronization failed: {checks}")
        checks.update({
            "model_state_sha256": state_fingerprint(model_state),
            "optimizer_state_sha256": state_fingerprint(optimizer_state),
        })
        return checks

    def run_pair(
        self,
        keras_model_state: dict[str, np.ndarray], keras_optimizer_state: dict[str, np.ndarray], keras_step: int,
        torch_model_state: dict[str, np.ndarray], torch_optimizer_state: dict[str, np.ndarray], torch_step: int,
    ) -> dict[str, object]:
        tf, torch = self.tf, self.torch
        if keras_step != torch_step:
            raise RuntimeError("Probe optimizer steps must match")
        load_keras_weights(self.keras_model, keras_model_state)
        load_torch_weights(self.torch_model, torch_model_state)
        keras_adam = adam_from_checkpoint(keras_model_state, keras_optimizer_state, keras_step)
        torch_adam = adam_from_checkpoint(torch_model_state, torch_optimizer_state, torch_step)

        before_keras = export_keras_weights(self.keras_model)
        before_torch = export_torch_weights(self.torch_model)
        before_weight = compare_group(before_keras, before_torch, TRAINABLE_NAMES)
        before_m = compare_group(adam_m_state(keras_adam), adam_m_state(torch_adam), TRAINABLE_NAMES)
        before_v = compare_group(adam_v_state(keras_adam), adam_v_state(torch_adam), TRAINABLE_NAMES)
        before_bn_mean = compare_group(before_keras, before_torch, BN_MEAN_NAMES)
        before_bn_variance = compare_group(before_keras, before_torch, BN_VARIANCE_NAMES)

        with tf.GradientTape() as tape:
            keras_logits = self.keras_model(self.kx, training=True)
            keras_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    self.ky, keras_logits, from_logits=True,
                )
            )
        keras_gradient = keras_gradients(
            self.keras_model, tape.gradient(keras_loss, self.keras_model.trainable_variables),
        )
        keras_after_forward = export_keras_weights(self.keras_model)
        keras_updated, keras_update = keras_adam.update(
            trainable_values(keras_after_forward), keras_gradient,
        )
        load_keras_trainable(self.keras_model, keras_updated)
        keras_post = export_keras_weights(self.keras_model)

        self.torch_model.train()
        self.torch_model.zero_grad(set_to_none=True)
        torch_logits = self.torch_model(self.px)
        torch_loss = torch.nn.functional.cross_entropy(torch_logits, self.py)
        torch_loss.backward()
        torch_gradient = torch_gradients(self.torch_model)
        torch_after_forward = export_torch_weights(self.torch_model)
        torch_updated, torch_update = torch_adam.update(
            trainable_values(torch_after_forward), torch_gradient,
        )
        load_torch_trainable(self.torch_model, torch_updated)
        torch_post = export_torch_weights(self.torch_model)

        for values, name in (
            (keras_gradient, "Keras gradient"), (torch_gradient, "PyTorch gradient"),
            (keras_update, "Keras update"), (torch_update, "PyTorch update"),
            (keras_post, "Keras post-step state"), (torch_post, "PyTorch post-step state"),
        ):
            assert_finite(values, name)

        logits_stats = compare(
            keras_logits.numpy(), torch_logits.detach().cpu().numpy(),
        )
        gradient_stats = compare_group(keras_gradient, torch_gradient, TRAINABLE_NAMES)
        update_stats = compare_group(keras_update, torch_update, TRAINABLE_NAMES)
        post_weight = compare_group(keras_post, torch_post, TRAINABLE_NAMES)
        post_m = compare_group(adam_m_state(keras_adam), adam_m_state(torch_adam), TRAINABLE_NAMES)
        post_v = compare_group(adam_v_state(keras_adam), adam_v_state(torch_adam), TRAINABLE_NAMES)
        post_bn_mean = compare_group(keras_post, torch_post, BN_MEAN_NAMES)
        post_bn_variance = compare_group(keras_post, torch_post, BN_VARIANCE_NAMES)
        before = float(before_weight["relative_l2_error"])
        after = float(post_weight["relative_l2_error"])
        return {
            "optimizer_step": keras_step,
            "loss_keras": float(keras_loss.numpy()),
            "loss_pytorch": float(torch_loss.detach().cpu()),
            "loss_abs_diff": abs(float(keras_loss.numpy()) - float(torch_loss.detach().cpu())),
            "logits_max_abs_diff": logits_stats["max_abs_diff"],
            "logits_relative_l2": logits_stats["relative_l2_error"],
            "gradient_relative_l2": gradient_stats["relative_l2_error"],
            "gradient_cosine": gradient_stats["cosine_similarity"],
            "gradient_max_abs_diff": gradient_stats["max_abs_diff"],
            "update_relative_l2": update_stats["relative_l2_error"],
            "before_weight_relative_l2": before,
            "post_weight_relative_l2": after,
            "weight_divergence_delta": after - before,
            "weight_growth_ratio": "" if before <= 1e-12 else after / before,
            "before_adam_m_relative_l2": before_m["relative_l2_error"],
            "before_adam_v_relative_l2": before_v["relative_l2_error"],
            "adam_m_relative_l2": post_m["relative_l2_error"],
            "adam_v_relative_l2": post_v["relative_l2_error"],
            "before_bn_mean_relative_l2": before_bn_mean["relative_l2_error"],
            "before_bn_variance_relative_l2": before_bn_variance["relative_l2_error"],
            "bn_mean_relative_l2": post_bn_mean["relative_l2_error"],
            "bn_variance_relative_l2": post_bn_variance["relative_l2_error"],
            "nan_or_inf": False,
            "probe_config_sha256": config_hash(),
        }


def expected_resync_cases() -> int:
    return len(SEEDS) * sum(len(anchors(checkpoint)) for checkpoint in CHECKPOINTS)
