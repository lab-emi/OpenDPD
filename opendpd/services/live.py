"""Bounded, display-only observations of the existing training/evaluation pipeline.

The legacy Project, loaders and optimizer still do all scientific work. The
observer wraps an instance's loader; it never samples its shuffled iterator in
advance. A separate frozen network previews a fixed validation segment. Neither
its output nor its metrics participates in checkpoint selection.
"""
from __future__ import annotations

import copy
import json
import math
import time
from datetime import datetime, timezone

from opendpd.core.plots import spectrum, time_excerpt
from opendpd.schemas import RunEventType
from opendpd.services.workspace import write_json_atomic

LIVE_FILE = "live.json"
POLICY = {"min_batches": 25, "min_seconds": 2.0, "progress_seconds": 0.5,
          "overhead_target": 0.05, "max_preview_samples": 16384}


def load_live(ws, run_id):
    path = ws.run_dir(run_id) / LIVE_FILE
    return json.loads(path.read_text()) if path.exists() else {"version": "live-v1", "policy": POLICY,
                                                            "geometry": None, "preview": None}


class ObservedLoader:
    def __init__(self, loader, observer, phase, after_batch=None):
        self.loader, self.observer, self.phase, self.after_batch = loader, observer, phase, after_batch

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        if self.phase == "train":
            self.observer.epoch += 1
        self.observer.stage(self.phase)
        for index, batch in enumerate(self.loader):
            yield batch                       # the original forward/backward/step runs here
            self.observer.batch(self.phase, index + 1, len(self), batch)
            if self.after_batch:
                self.after_batch(index + 1, len(self), batch)


class LiveMonitor:
    def __init__(self, ws, run_id, resolved, dataset, emit, *, clock=time.monotonic):
        self.ws, self.run_id, self.resolved, self.dataset, self.emit = ws, run_id, resolved, dataset, emit
        self.clock = clock
        self.epoch = -1
        self.steps = 0
        self.last_progress = -math.inf
        self.last_preview = -math.inf
        self.last_preview_step = 0
        self.preview_seconds = POLICY["min_seconds"]
        self.revision = 0
        self.shadow = None
        self.preview_error = None
        self.current = {}
        self.phase_samples = 0
        self.state = {"version": "live-v1", "policy": dict(POLICY), "geometry": None, "preview": None}

    def save(self):
        write_json_atomic(self.ws.run_dir(self.run_id) / LIVE_FILE, self.state)

    def stage(self, phase):
        self.phase_samples = 0
        self.current = {"scope": "live", "phase": phase, "epoch": max(0, self.epoch),
                        "total_epochs": self.resolved.training.epochs}
        self.emit(RunEventType.progress, dict(self.current))
        print(f"[OpenDPD] {phase} · {self.resolved.task.value}", flush=True)

    def batch(self, phase, index, total, batch):
        features = batch[0]
        if phase == "train":
            self.steps += 1
        self.current = {"scope": "live", "phase": phase, "epoch": max(0, self.epoch),
                        "total_epochs": self.resolved.training.epochs,
                        "batch": index, "total_batches": total, "global_step": self.steps,
                        "sequences": int(features.shape[0]), "sequence_samples": int(features.shape[1]),
                        "sample_rate_hz": self.dataset.signal.sample_rate_hz}
        split_name = "test" if phase == "evaluate" else phase
        tensor_samples = int(features.shape[0] * features.shape[1])
        if split_name in ("val", "test"):
            version = self.dataset.version(self.resolved.dataset.preprocessing_version)
            split = version.split if version is not None else self.dataset.split
            boundary = (split.boundaries or {}).get(split_name)
            if boundary:
                valid = min(tensor_samples, max(0, int(boundary[1] - boundary[0]) - self.phase_samples))
                self.current.update(valid_samples=valid, padded_samples=tensor_samples - valid)
        self.phase_samples += tensor_samples
        self.state["last_batch"] = dict(self.current)
        now = self.clock()
        if index == 1 or index == total or now - self.last_progress >= POLICY["progress_seconds"]:
            self.last_progress = now
            self.emit(RunEventType.progress, dict(self.current))
            print(f"[OpenDPD] {phase} epoch {max(0, self.epoch) + 1} · batch {index}/{total} · "
                  f"{features.shape[0]} I/Q sequences × {features.shape[1]} I/Q samples "
                  f"@ {self.dataset.signal.sample_rate_hz or 'unknown'} Hz", flush=True)

    def observe_checkpoints(self, project):
        from pathlib import Path
        from opendpd.services.model_download import MAX_MODEL_BYTES, publish_model
        save_best = project.logger.save_best_model
        last_checkpoint = None

        def save_checkpoint(*args, **kwargs):
            nonlocal last_checkpoint
            result = save_best(*args, **kwargs)
            source = Path(project.path_save_file_best)
            if source.is_file():
                stamp = (source.stat().st_mtime_ns, source.stat().st_size)
                if stamp != last_checkpoint and stamp[1] <= MAX_MODEL_BYTES:
                    publish_model(self.ws.run_dir(self.run_id), source.read_bytes(), epoch=max(0, self.epoch) + 1)
                    last_checkpoint = stamp
                    self.emit(RunEventType.checkpoint, {'epoch': max(0, self.epoch), 'available': True})
            return result

        project.logger.save_best_model = save_checkpoint

    def attach(self, project):
        original = project.train

        def train(**kwargs):
            self.observe_checkpoints(project)
            loader = kwargs["train_loader"]
            self.state["geometry"] = {
                "batch_size": loader.batch_size, "sequence_samples": project.frame_length,
                "frame_stride": project.frame_stride, "train_sequences": len(loader.dataset),
                "batches_per_epoch": len(loader), "sample_rate_hz": self.dataset.signal.sample_rate_hz,
                "eval_batch_size": kwargs["val_loader"].batch_size,
                "eval_sequence_samples": project.args.nperseg,
            }
            self.save()
            net = kwargs["net"]
            val_set = kwargs["val_loader"].dataset

            def preview(*_):
                now = self.clock()
                if (self.steps - self.last_preview_step < POLICY["min_batches"] or
                    now - self.last_preview < self.preview_seconds) and self.revision > 0:
                    return
                if self.preview_error or not len(val_set):
                    return
                start = self.clock()
                try:
                    import torch
                    # No DataLoader iterator: constructing one consumes the global torch RNG.
                    x, target = val_set[0]
                    if len(x) > POLICY["max_preview_samples"]:
                        raise ValueError("validation segment exceeds the live preview sample budget")
                    devices = [project.device.index or 0] if project.device.type == "cuda" else []
                    with torch.random.fork_rng(devices=devices), torch.inference_mode():
                        if self.shadow is None:
                            self.shadow = copy.deepcopy(net).eval().requires_grad_(False)
                        self.shadow.load_state_dict(net.state_dict())
                        prediction = self.shadow(x.unsqueeze(0).to(project.device)).cpu().numpy()
                    self.publish(prediction, target.unsqueeze(0).numpy(), x.unsqueeze(0).numpy(),
                                 project.args, "validation_probe")
                    self.last_preview = self.clock()
                    self.last_preview_step = self.steps
                    # Adapt to slow models; display work targets <=5% of elapsed time.
                    # The first sample includes one-time imports/copy/kernel warm-up.
                    if self.revision > 1:
                        self.preview_seconds = max(POLICY["min_seconds"],
                                                   (self.clock() - start) / POLICY["overhead_target"])
                except Exception as error:
                    self.preview_error = f"{type(error).__name__}: {error}"
                    self.state["preview_error"] = self.preview_error
                    self.save()
                    print(f"[OpenDPD] Live signal preview unavailable: {self.preview_error}", flush=True)

            kwargs["train_loader"] = ObservedLoader(loader, self, "train", preview)
            for phase in ("val", "test"):
                kwargs[f"{phase}_loader"] = ObservedLoader(kwargs[f"{phase}_loader"], self, phase)
            try:
                return original(**kwargs)
            finally:
                self.shadow = None

        project.train = train

    def evaluation_loader(self, loader, net, project):
        """Observe the already-computed first test batch; no extra model call."""
        if self.state["geometry"] and "frame_stride" in self.state["geometry"]:
            self.state["training_geometry"] = self.state["geometry"]
        self.state["geometry"] = {"batch_size": loader.batch_size,
                                  "sequence_samples": project.args.nperseg,
                                  "sample_rate_hz": self.dataset.signal.sample_rate_hz,
                                  "batches_per_epoch": len(loader), "train_sequences": len(loader.dataset)}
        self.save()
        captured = {}
        handle = net.register_forward_hook(lambda _net, _args, output: captured.update(output=output.detach()))

        def after(index, total, batch):
            if index != 1 or "output" not in captured:
                return
            handle.remove()
            # A streaming variant's offline bootstrap is not its final execution.
            if self.resolved.model.key.endswith("_stream"):
                return
            x, target = batch
            if self.resolved.task.value in ("train_dpd", "run_dpd"):
                target = project.target_gain * x
            try:
                self.publish(captured.pop("output")[:1].cpu().numpy(), target[:1].numpy(), x[:1].numpy(),
                             project.args, "test_probe")
            except Exception as error:
                print(f"[OpenDPD] Test preview unavailable: {error}", flush=True)

        return ObservedLoader(loader, self, "evaluate", after), handle

    def publish(self, prediction, target, x, args, source):
        from utils.metrics import NMSE, ACLR
        from opendpd.core.metrics import get_profile
        values = {"NMSE": float(NMSE(prediction, target))}
        dpd = self.resolved.task.value in ("train_dpd", "run_dpd")
        if dpd:
            left, right = ACLR(prediction, fs=args.input_signal_fs, nperseg=args.nperseg,
                               bw_main_ch=args.bw_main_ch, n_sub_ch=args.n_sub_ch)
            values.update(ACLR_L=float(left), ACLR_R=float(right), ACLR_AVG=float((left + right) / 2))
        values = {key: value for key, value in values.items() if math.isfinite(value)}
        signals = {"Input x": x, "Linear target" if dpd else "Measured PA output": target,
                   "DPD → PA surrogate" if dpd else "PA model output": prediction}
        roles = dict(zip(signals, ("input", "reference", "primary")))
        spec = self.dataset.signal
        plots = {"time": time_excerpt(signals, roles),
                 "spectrum": spectrum(signals, roles, sample_rate_hz=spec.sample_rate_hz,
                                      nperseg=spec.nperseg, bandwidth_hz=spec.bandwidth_hz)}
        self.revision += 1
        units = {key: get_profile("legacy-opendpd-v1").metric(key).unit for key in values}
        preview = {"revision": self.revision, "updated_at": datetime.now(timezone.utc).isoformat(),
                   "source": source, "epoch": max(0, self.epoch), "global_step": self.steps,
                   "metrics": values, "units": units,
                   "metric_profile": "legacy-opendpd-v1",
                   "plots": plots, "samples": int(prediction.shape[1]),
                   "interval_seconds": self.preview_seconds}
        self.state["preview"] = preview
        self.save()
        position = max(0, self.epoch) + self.current.get("batch", 0) / max(1, self.current.get("total_batches", 1))
        self.emit(RunEventType.metric, {"epoch": position, "split": source, "values": values})
        self.emit(RunEventType.progress, {**self.current, "preview_revision": self.revision})
        print(f"[OpenDPD] {source} · " + " · ".join(f"{k}={v:.4f} {units[k]}" for k, v in values.items()), flush=True)

    def complete(self, result):
        plots = {}
        for kind in ("time", "spectrum"):
            path = self.ws.run_dir(self.run_id) / "plots" / f"{kind}.json"
            if path.exists():
                plots[kind] = json.loads(path.read_text())
        self.revision += 1
        values = {m.name: m.value for m in result.metrics if m.value is not None}
        self.state["preview"] = {"revision": self.revision, "updated_at": datetime.now(timezone.utc).isoformat(),
                                 "source": "final_test", "metrics": values,
                                 "units": {m.name: m.unit for m in result.metrics}, "plots": plots,
                                 "metric_profile": result.metric_profile_id,
                                 "samples": (result.valid_sample_range[1] - result.valid_sample_range[0])
                                 if result.valid_sample_range else None}
        self.save()
        self.emit(RunEventType.progress, {"scope": "live", "phase": "complete", "preview_revision": self.revision})
        print(f"[OpenDPD] Final test · {result.metric_profile_id} · " +
              " · ".join(f"{m.name}={m.value:.4f} {m.unit}" for m in result.metrics if m.value is not None), flush=True)
