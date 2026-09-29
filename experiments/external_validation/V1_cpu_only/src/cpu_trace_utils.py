"""CPU-only adapters around the read-only Experiment 07/08 machinery."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EXP07 = ROOT / "experiments/07_multistep_state_resynchronization"
EXP08 = ROOT / "experiments/08_layer_by_layer_trajectory"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP07))
sys.path.insert(0, str(EXP08))

from layer_trace import LayerTraceRunner, STANDARD_FORWARD_STAGES, _numpy
from probe_utils import ProbeRunner

from common.common_adam_control import (
    keras_gradients, load_keras_trainable, load_torch_trainable, torch_gradients,
    trainable_values,
)
from common.controlled_batchnorm import build_keras_common_bn_model, build_torch_common_bn_model
from common.controlled_initialization import (
    export_keras_weights, export_torch_weights, load_keras_weights, load_torch_weights,
)
from common.state_resynchronization import adam_from_checkpoint, adam_m_state, adam_v_state
from v1_config import (
    DTYPE, TF_INTER_OP_THREADS, TF_INTRA_OP_THREADS, TORCH_INTER_OP_THREADS,
    TORCH_NUM_THREADS, config_hash,
)

_RUNTIME = None


def configure_cpu_runtime():
    """Hide TF accelerators and pin both frameworks to one CPU thread."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        raise RuntimeError("TensorFlow was initialized before CPU-only device masking") from error
    tf.config.threading.set_intra_op_parallelism_threads(TF_INTRA_OP_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(TF_INTER_OP_THREADS)
    import torch
    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.set_num_interop_threads(TORCH_INTER_OP_THREADS)
    _RUNTIME = (tf, torch)
    return _RUNTIME


def actual_device_check() -> dict[str, object]:
    tf, torch = configure_cpu_runtime()
    with tf.device("/CPU:0"):
        tf_value = tf.add(tf.constant([1.0]), tf.constant([2.0]))
    torch_value = torch.tensor([1.0], device=torch.device("cpu")) + 2.0
    tf_device = str(tf_value.device)
    torch_device = str(torch_value.device)
    return {
        "tensorflow_actual_device": tf_device,
        "pytorch_actual_device": torch_device,
        "tensorflow_cpu_actual": "CPU:0" in tf_device.upper(),
        "pytorch_cpu_actual": torch_value.device.type == "cpu",
        "tensorflow_visible_gpu_count": len(tf.config.get_visible_devices("GPU")),
        "tensorflow_logical_devices": [device.name for device in tf.config.list_logical_devices()],
        "pytorch_mps_tensor_used": torch_value.device.type == "mps",
    }


def _command_value(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        return completed.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest() -> dict[str, object]:
    tf, torch = configure_cpu_runtime()
    devices = actual_device_check()
    try:
        metal_version = importlib.metadata.version("tensorflow-metal")
    except importlib.metadata.PackageNotFoundError:
        metal_version = None
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception as error:  # environment reporting must preserve the failure text
        torchvision_version = f"unavailable: {type(error).__name__}: {error}"
    try:
        import keras
        keras_version = keras.__version__
    except Exception:
        keras_version = getattr(tf.keras, "__version__", "unknown")
    cpu_model = _command_value(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    return {
        "study": "External Validation Study",
        "validation": "V1 - CPU-only Execution Validation",
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "keras_version": keras_version,
        "tensorflow_metal_installed": metal_version is not None,
        "tensorflow_metal_version": metal_version,
        "tensorflow_physical_devices": [device.name for device in tf.config.list_physical_devices()],
        "tensorflow_visible_devices": [device.name for device in tf.config.get_visible_devices()],
        "tensorflow_actual_execution_device": devices["tensorflow_actual_device"],
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "pytorch_mps_available": bool(torch.backends.mps.is_available()),
        "pytorch_actual_execution_device": devices["pytorch_actual_device"],
        "numpy_version": np.__version__,
        "cpu_model": cpu_model,
        "thread_settings": {
            "tensorflow_intra_op": TF_INTRA_OP_THREADS,
            "tensorflow_inter_op": TF_INTER_OP_THREADS,
            "pytorch_num_threads": torch.get_num_threads(),
            "pytorch_inter_op_threads": torch.get_num_interop_threads(),
        },
        "float_dtype": DTYPE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _command_value(["git", "rev-parse", "HEAD"]),
        "config_sha256": config_hash(),
        "run_full_training": False,
    }


class CpuLayerTraceRunner(LayerTraceRunner):
    """Experiment 08 trace runner with both frameworks explicitly on CPU."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        tf, torch = configure_cpu_runtime()
        self.tf, self.torch = tf, torch
        tf.keras.backend.set_floatx("float32")
        self.device = torch.device("cpu")
        self.images = np.ascontiguousarray(images.astype(np.float32, copy=False))
        self.labels = np.asarray(labels, dtype=np.int32)
        with tf.device("/CPU:0"):
            self.kx = tf.convert_to_tensor(self.images, dtype=tf.float32)
            self.ky = tf.convert_to_tensor(self.labels, dtype=tf.int32)
            self.keras_model = build_keras_common_bn_model()
            keras_outputs = []
            for name in STANDARD_FORWARD_STAGES:
                layer_name = "dense128_relu" if name == "fc128_relu" else name
                keras_outputs.append(self.keras_model.get_layer(layer_name).output)
            self.keras_trace_model = tf.keras.Model(self.keras_model.input, keras_outputs)
            self.plain_keras_model = build_keras_common_bn_model()
        self.px = torch.from_numpy(self.images.transpose(0, 3, 1, 2).copy()).to(self.device)
        self.py = torch.from_numpy(self.labels.astype(np.int64)).to(self.device)
        restored = self.px.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.input_exact = bool(np.array_equal(self.images, restored))
        self.label_exact = bool(np.array_equal(self.labels, self.py.detach().cpu().numpy().astype(np.int32)))
        if not self.input_exact or not self.label_exact:
            raise RuntimeError("CPU canonical input/label conversion mismatch")
        self.torch_model = build_torch_common_bn_model().to(self.device)
        self.plain_torch_model = build_torch_common_bn_model().to(self.device)
        self.torch_activations = {}
        self._hook_handles = []
        for name in STANDARD_FORWARD_STAGES:
            module_name = "dense128_relu" if name == "fc128_relu" else name
            module = getattr(self.torch_model, module_name)
            self._hook_handles.append(module.register_forward_hook(self._make_hook(name)))

    def _plain_keras_step(self, model_state, optimizer_state, step):
        tf = self.tf
        load_keras_weights(self.plain_keras_model, model_state)
        adam = adam_from_checkpoint(model_state, optimizer_state, step)
        with tf.device("/CPU:0"):
            with tf.GradientTape() as tape:
                logits = self.plain_keras_model(self.kx, training=True)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    self.ky, logits, from_logits=True,
                ))
            gradients = keras_gradients(
                self.plain_keras_model,
                tape.gradient(loss, self.plain_keras_model.trainable_variables),
            )
        after_forward = export_keras_weights(self.plain_keras_model)
        updated, updates = adam.update(trainable_values(after_forward), gradients)
        load_keras_trainable(self.plain_keras_model, updated)
        return {
            "logits": _numpy(logits), "loss": float(loss.numpy()),
            "parameter_gradients": gradients, "updates": updates,
            "adam_m": adam_m_state(adam), "adam_v": adam_v_state(adam),
            "post": export_keras_weights(self.plain_keras_model),
        }

    def _plain_torch_step(self, model_state, optimizer_state, step):
        torch = self.torch
        load_torch_weights(self.plain_torch_model, model_state)
        adam = adam_from_checkpoint(model_state, optimizer_state, step)
        self.plain_torch_model.train()
        self.plain_torch_model.zero_grad(set_to_none=True)
        logits = self.plain_torch_model(self.px)
        loss = torch.nn.functional.cross_entropy(logits, self.py)
        loss.backward()
        gradients = torch_gradients(self.plain_torch_model)
        after_forward = export_torch_weights(self.plain_torch_model)
        updated, updates = adam.update(trainable_values(after_forward), gradients)
        load_torch_trainable(self.plain_torch_model, updated)
        return {
            "logits": _numpy(logits), "loss": float(loss.detach().cpu()),
            "parameter_gradients": gradients, "updates": updates,
            "adam_m": adam_m_state(adam), "adam_v": adam_v_state(adam),
            "post": export_torch_weights(self.plain_torch_model),
        }

    def run_plain(self, model_state, optimizer_state, step):
        return (
            self._plain_keras_step(model_state, optimizer_state, step),
            self._plain_torch_step(model_state, optimizer_state, step),
        )
