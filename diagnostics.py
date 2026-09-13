"""Optional memory snapshots without resetting shared allocator statistics."""

import json
import logging
import os
import time
from pathlib import Path

import psutil
import torch


def log_memory(label, device):
    if os.environ.get("SELFLIFT_MEMORY_LOG", "0") != "1":
        return
    device = torch.device(device)
    fields = []
    try:
        fields.append(f"process_rss={psutil.Process().memory_info().rss / 2**20:.2f} MiB")
        fields.append(f"system_available={psutil.virtual_memory().available / 2**20:.2f} MiB")
    except (psutil.Error, OSError) as error:
        fields.append(f"host_memory_unavailable={error}")
    if device.type == "cuda" and torch.cuda.is_initialized():
        try:
            free, total = torch.cuda.mem_get_info(device)
            fields.extend([
                f"allocated={torch.cuda.memory_allocated(device) / 2**20:.2f} MiB",
                f"reserved={torch.cuda.memory_reserved(device) / 2**20:.2f} MiB",
                f"process_peak_allocated={torch.cuda.max_memory_allocated(device) / 2**20:.2f} MiB",
                f"device_used={((total - free) / 2**20):.2f} MiB",
                f"device_free={free / 2**20:.2f} MiB",
            ])
        except RuntimeError as error:
            fields.append(f"cuda_memory_unavailable={error}")
    logging.info("[SelfLift memory] %s device=%s; %s", label, device, "; ".join(fields))


def latent_detail_profile(video, sample_seconds=(0.5, 1.5, 3.5), fps=24.0):
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError("SelfLift latent diagnostics require [B,C,T,H,W] video")
    points = []
    for seconds in sample_seconds:
        frame = max(0, round(float(seconds) * float(fps)))
        latent_index = min(video.shape[2] - 1, 1 + frame * 5 // 17)
        sample = video[0, :, latent_index].detach().to(device="cpu", dtype=torch.float32)
        rms = sample.square().mean().sqrt()
        horizontal = (sample[..., 1:] - sample[..., :-1]).abs().mean()
        vertical = (sample[..., 1:, :] - sample[..., :-1, :]).abs().mean()
        points.append({
            "seconds": float(seconds),
            "latent_index": int(latent_index),
            "rms": float(rms),
            "spatial_gradient": float((horizontal + vertical) * 0.5),
            "normalized_spatial_gradient": float((horizontal + vertical) * 0.5 / rms.clamp_min(1e-8)),
        })
    return {"shape": list(video.shape), "samples": points}


def write_latent_diagnostics(stages, settings, enabled=None):
    if enabled is None:
        enabled = os.environ.get("SELFLIFT_LATENT_DIAGNOSTICS", "0") == "1"
    if not enabled:
        return None
    try:
        import folder_paths
        root = Path(folder_paths.get_output_directory())
    except ImportError:
        root = Path.cwd() / "output"
    directory = root / "selflift_diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"selflift_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {"settings": settings, "stages": {
        name: value if isinstance(value, dict) else latent_detail_profile(value)
        for name, value in stages.items() if value is not None
    }}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("[SelfLift diagnostics] wrote %s", path)
    return path
