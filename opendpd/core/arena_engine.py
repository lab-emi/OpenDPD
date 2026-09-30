"""Arena training and held-out cascade evaluation through one frozen PA.

Everything here determines trained weights or raw judge observations, so this
file belongs to the training fingerprint. Orchestration, presets, the cost
model and scoring live in arena_runner/arena/arena_ops and can change without
invalidating a checkpoint.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from models import CascadedModel, CoreModel
from opendpd.core import arena
from opendpd.core.arena_metrics import compute as compute_metrics, validation_metrics
from opendpd.core.polynomial import (
    PolynomialModel,
    basis,
    fit_least_squares,
    to_complex,
)
from opendpd.core.registry import get_model, validate_parameters
from opendpd.schemas import SignalSpec
from opendpd.services.workspace import read_json, write_json_atomic


def _disable_statistics(model):
    for module in model.modules():
        if hasattr(module, "set_debug"):
            module.set_debug(0)
        if hasattr(module, "debug"):
            module.debug = False
    return model


def build_model(key, parameters):
    descriptor = get_model(key)
    base = descriptor.weights_from or key
    params = validate_parameters(base, parameters, "dpd")
    if base in arena.DETERMINISTIC:
        return PolynomialModel(base, params)
    net = CoreModel(
        input_size=2,
        hidden_size=int(params.get("hidden_size", 23)),
        num_layers=int(params.get("num_layers", 1)),
        backbone_type=base,
        window_size=4,
        num_dvr_units=int(params.get("num_dvr_units", 3)),
        thx=float(params.get("thx", 0)),
        thh=float(params.get("thh", 0)),
        user_definition=params.get("definition"),
    )
    from opendpd.core.arena_training_backends import enable
    return enable(_disable_statistics(net))


def _asset(file, expected):
    if not isinstance(file, str) or Path(file).name != file:
        raise ValueError("Invalid packaged Arena asset")
    path = arena.ASSETS / file
    if not path.is_file() or path.is_symlink() or arena.file_hash(path) != expected:
        raise ValueError(f"Arena asset failed integrity check: {file}")
    return path


def _load_weights(model, path):
    with np.load(path, allow_pickle=False) as arrays:
        state = {
            key: torch.from_numpy(np.array(arrays[key], copy=True))
            for key in arrays.files
        }
    model.load_state_dict(state, strict=True)
    return model


def _save_weights(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        **{
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        },
    )
    temporary.replace(path)


def load_frozen(spec, device):
    model = build_model(spec["model"]["key"], spec["model"]["parameters"])
    _load_weights(model, _asset(spec["file"], spec["sha256"]))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def limit_tensor(iq, peak):
    amp = torch.linalg.vector_norm(iq, dim=-1, keepdim=True)
    return iq * torch.clamp(peak / amp.clamp_min(1e-12), max=1.0)


def limit_array(iq, peak):
    amplitude = np.linalg.norm(iq, axis=-1, keepdims=True)
    return np.asarray(
        iq * np.minimum(peak / np.maximum(amplitude, 1e-12), 1.0), dtype=np.float32
    )


class ArenaCascade(CascadedModel):
    """Fixed transmitter limiter and shared training loss context."""

    _arena_static_cascade = True

    def __init__(self, dpd_model, pa_model, peak):
        super().__init__(dpd_model, pa_model)
        self.peak = float(peak)
        self.freeze_pa_model()

    def forward(self, x):
        return self.pa_model(limit_tensor(self.dpd_model(x), self.peak))[:, 50:150]


def offline_output(model, iq, device="cpu", batch_size=64):
    """One continuous output from a declared overlap-save DPD context."""
    n = len(iq)
    count = (n + 99) // 100
    padded = np.pad(np.asarray(iq, dtype=np.float32), ((50, 150 + (-n) % 100), (0, 0)))
    pieces = []
    model.eval()
    # no_grad also permits training again after validation in legacy Delta
    # modules which retain diagnostic tensors; inference_mode does not.
    with torch.no_grad():
        for first in range(0, count, batch_size):
            starts = range(first * 100, min(count, first + batch_size) * 100, 100)
            window = np.stack([padded[start : start + 200] for start in starts])
            output = model(torch.from_numpy(window).to(device))[:, 50:150]
            pieces.append(output.detach().cpu().numpy().reshape(-1, 2))
    return np.concatenate(pieces)[:n]


def dpd_output(model, key, iq, device="cpu"):
    if get_model(key).weights_from:
        from opendpd.core.streaming import run_stream
        from opendpd.services.streaming import streaming_model

        model.to("cpu").eval()
        return run_stream(
            streaming_model(key, model), np.asarray(iq, dtype=np.float32), 200
        )
    return offline_output(model, iq, device)


def pa_output(model, iq, device="cpu"):
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(iq, dtype=np.float32)).unsqueeze(0).to(device)
        if len(iq) <= 32768:
            return model(x)[0].cpu().numpy()
        # cuDNN limits long recurrent sequence descriptors. Preserve the full
        # TRes context and GRU state while splitting only recurrent execution.
        from backbones.tres_gru import TResGRU
        from backbones.finite_iq import amplitude
        backbone = getattr(model, "backbone", None)
        if not isinstance(backbone, TResGRU):
            return model(x)[0].cpu().numpy()
        lookahead = torch.roll(x, shifts=-1, dims=1)
        radius = amplitude(torch.sum(x*x, dim=-1, keepdim=True))
        features = torch.cat((x, radius, radius**3, lookahead), dim=-1)
        skip = backbone.tcn(x.transpose(1, 2)).transpose(1, 2)
        pieces, state = [], None
        for first in range(0, len(iq), 16384):
            recurrent, state = backbone.rnn(features[:, first:first+16384], state)
            pieces.append(backbone.fc_out(recurrent) + skip[:, first:first+16384])
        return torch.cat(pieces, dim=1)[0].cpu().numpy()


def signal_metrics(y, reference, signal, grid, start, window=None):
    if not np.isfinite(y).all():
        raise FloatingPointError("The candidate produced nonfinite PA output")
    return compute_metrics(y, reference, signal, grid, start, window)


def _complex_scale(iq, value):
    z = to_complex(iq) * value
    return np.stack((z.real, z.imag), axis=-1).astype(np.float32)


def calibrate_linear_baseline(teacher, x, gain, peak, device):
    """One common input rule, calibrated on TRAIN through the public teacher."""
    probe = x[:4096]
    y = pa_output(teacher, probe, device)
    beta = np.vdot(to_complex(probe), to_complex(y)) / np.vdot(
        to_complex(probe), to_complex(probe)
    )
    phase = np.exp(-1j * np.angle(beta))
    target_energy = float(np.mean(np.square((gain * probe).astype(np.float64))))
    lo, hi = 0.01, 2.0
    for _ in range(18):
        mid = (lo + hi) / 2
        pred = pa_output(
            teacher, limit_array(_complex_scale(probe, mid * phase), peak), device
        )
        if float(np.mean(np.square(pred.astype(np.float64)))) < target_energy:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 * phase


class Condition:
    def __init__(self, identifier, device):
        self.identifier, self.device = identifier, device
        self.manifest = arena.calibration()[identifier]
        m = self.manifest
        with np.load(
            _asset(m["data_file"], m["data_sha256"]), allow_pickle=False
        ) as data:
            self.data = {
                key: np.asarray(data[key], dtype=np.float32) for key in data.files
            }
        self.signal = SignalSpec.model_validate(m["signal"])
        self.grid = m["evm_grid"]
        self.starts = {name: span[0] for name, span in m["split"]["boundaries"].items()}
        x, y = self.data["x_train"], self.data["y_train"]
        self.peak = float(m["peak_limit"])
        self.gain = float(m["reference_gain"])
        self.teacher = load_frozen(m["teacher"], device)
        self.alpha = calibrate_linear_baseline(
            self.teacher, x, self.gain, self.peak, device
        )
        self._baselines = None

    def outputs(self, iq):
        yield "pa", self.manifest["teacher"]["sha256"], pa_output(self.teacher, iq, self.device)

    def metrics(self, y, split):
        """Waveform metrics of one PA output against the fixed g·x target of a split."""
        window = self.manifest["stimuli"]["splits"][split]
        first, last = window["metric_start"], window["metric_stop"]
        return signal_metrics(y, self.gain * self.data[f"x_{split}"],
                              self.signal, self.grid, self.starts[split], (first, last))

    def baseline_metrics(self):
        if self._baselines is None:
            x = self.data["x_test"]
            u0 = limit_array(_complex_scale(x, self.alpha), self.peak)
            self._baselines = {
                name: self.metrics(y, "test") for name, _, y in self.outputs(u0)
            }
        return self._baselines


def _seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def training_frames(iq, device):
    """A view of all valid windows; no duplication of the whole capture on GPU."""
    samples = torch.from_numpy(np.ascontiguousarray(iq, dtype=np.float32)).to(device)
    return samples.unfold(0, arena.TRAINING["frame_length"], arena.TRAINING["frame_stride"]).transpose(1, 2)


def validation_objective(condition, prediction):
    """Short-record spectral selection on validation only, with explicit bands."""
    from opendpd.core.metrics.spectral_v2 import compute
    reference = condition.gain * condition.data["x_val"]
    scores = validation_metrics(prediction, reference)
    signal = SignalSpec.model_validate(condition.manifest["signal"]).model_copy(
        update={"nperseg": arena.TRAINING["validation_nperseg"]})
    context = arena.TRAINING["validation_context"]
    window = slice(context, len(reference) - context)
    band = {m.name: m.value for m in compute(prediction[window], reference[window], signal)}
    if any(band.get(key) is None for key in ("IBE", "ACLR_L", "ACLR_R")):
        raise ValueError("Validation capture does not support the declared spectral estimator")
    scores.update(ib_error_db=band["IBE"], aclr_l_db=band["ACLR_L"], aclr_r_db=band["ACLR_R"])
    scores["objective_db"] = .5 * (band["IBE"] + max(band["ACLR_L"], band["ACLR_R"]))
    return scores


def train_gradient(condition, key, parameters, seed, folder, emit, *, ready=None):
    _seed(seed)
    device = condition.device
    model = build_model(key, parameters).to(device)
    net = ArenaCascade(model, condition.teacher, condition.peak).to(device)
    trainable = list(model.parameters())
    if sum(p.numel() for p in trainable) > arena.TRAINING["max_parameters"]:
        raise ValueError("Backbone exceeds the 4,096-parameter Arena limit")
    optimizer = torch.optim.AdamW(trainable, lr=arena.TRAINING["learning_rate"],
                                  weight_decay=arena.TRAINING["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=arena.TRAINING["lr_factor"],
        patience=arena.TRAINING["lr_patience_validation_checks"], min_lr=arena.TRAINING["lr_end"],
        threshold_mode=arena.TRAINING["lr_threshold_mode"], threshold=arena.TRAINING["lr_threshold_db"],
    )
    criterion = torch.nn.MSELoss()
    from opendpd.core.arena_cuda import make_arena_step

    step = (
        make_arena_step(net, criterion, trainable, arena.TRAINING["gradient_clip"]) if device == "cuda" else None
    )
    rng = np.random.default_rng(seed)
    x = condition.data["x_train"]
    xv = condition.data["x_val"]
    history, draws = [], hashlib.sha256()
    best, best_epoch, best_state = (False, -math.inf), None, None
    start_time = time.monotonic()
    epochs = arena.TRAINING["epochs"]
    frames = training_frames(x, device)
    budget = arena.training_budget(len(x))
    batch = arena.TRAINING["batch_size"]
    updates, first_epoch, prior_seconds = 0, 1, 0.
    resume_path = folder / "resume.pt"
    binding = cache_binding(key, parameters, condition.identifier, seed)
    if resume_path.exists():
        resumed = torch.load(resume_path, map_location=device, weights_only=True)
        if resumed["binding"] != binding or resumed["device"] != device:
            raise ValueError("Resume state does not match the frozen training request/device")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        history = resumed["history"]
        best, best_epoch, best_state = resumed["best"], resumed["best_epoch"], resumed["best_state"]
        updates, first_epoch, prior_seconds = resumed["updates"], resumed["epoch"] + 1, resumed["seconds"]
        for _ in range(first_epoch - 1):
            draws.update(rng.permutation(len(frames)).astype("<i8").tobytes())
        torch.set_rng_state(resumed["rng_cpu"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state(resumed["rng_cuda"].cpu())
    if ready is not None:
        # Initialization and CUDA capture are serialized by the caller. Warm
        # both batch shapes before releasing the seed barrier: no compilation
        # or capture may start while the other seeds are replaying graphs.
        if step is None:
            raise RuntimeError("Parallel Arena seeds require a reviewed CUDA replay path")
        net.train()
        warm = frames[:batch]
        if step(warm, condition.gain * warm[:, 50:150]) is None:
            raise RuntimeError("Parallel Arena CUDA capture was unavailable")
        tail = len(frames) % batch
        if tail:
            optimizer.zero_grad(set_to_none=False)
            short = frames[:tail]
            criterion(net(short), condition.gain * short[:, 50:150]).backward()
            torch.nn.utils.clip_grad_norm_(trainable, arena.TRAINING["gradient_clip"])
        step(warm, condition.gain * warm[:, 50:150])
        ready()
    for epoch in range(first_epoch, epochs + 1):
        order = rng.permutation(len(frames)).astype("<i8")
        draws.update(order.tobytes())
        order = torch.from_numpy(order).to(device)
        net.train()
        losses = []
        for first in range(0, len(frames), batch):
            features = frames[order[first : first + batch]]
            target = condition.gain * features[:, 50:150]
            loss = step(features, target) if step else None
            if loss is None:
                optimizer.zero_grad(set_to_none=step is None)
                loss = criterion(net(features), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, arena.TRAINING["gradient_clip"])
            optimizer.step()
            updates += 1
            losses.append(loss.detach() * len(features))
        losses = torch.stack(losses)
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
        if epoch % arena.TRAINING["validation_every_epochs"] == 0:
            u = limit_array(offline_output(model, xv, device), condition.peak)
            y = pa_output(condition.teacher, u, device)
            scores = validation_objective(condition, y)
            feasible = (
                abs(scores["power_error_db"])
                <= arena.TRAINING["output_power_tolerance_db"]
            )
            objective = (feasible, -scores["objective_db"])
            scheduler.step(scores["objective_db"])
            if objective > best:
                best, best_epoch = objective, epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            history.append(
                dict(
                    epoch=epoch,
                    loss=float(losses.sum().cpu()) / len(frames),
                    optimizer_updates=updates,
                    validation=scores,
                    validation_nmse_db=scores["nmse_db"],
                    validation_objective_db=scores["objective_db"],
                    validation_feasible=feasible,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )
            )
            emit(
                "training",
                epoch,
                f"{condition.identifier} · seed {seed} · epoch {epoch}/{epochs}",
            )
            write_json_atomic(folder / "history.json", history)
        temporary = resume_path.with_suffix(".tmp.pt")
        torch.save(dict(binding=binding, device=device, epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), history=history,
            best=best, best_epoch=best_epoch, best_state=best_state, updates=updates,
            seconds=prior_seconds + time.monotonic() - start_time, rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state() if device == "cuda" else None), temporary)
        temporary.replace(resume_path)
    if best_state is None:
        raise RuntimeError("No valid validation checkpoint")
    model.load_state_dict(best_state)
    del step, optimizer, net
    model.eval()
    _save_weights(model, folder / "weights.npz")
    info = dict(
        selected_epoch=best_epoch,
        attained_epochs=epochs,
        **budget,
        frame_draw_sha256=draws.hexdigest(),
        training_seconds=prior_seconds + time.monotonic() - start_time,
        training_device=device,
        selection=arena.TRAINING["checkpoint_selection"],
        selected_validation_feasible=bool(best[0]),
        selected_validation_objective_db=-best[1],
        selected_validation_nmse_db=next(h["validation_nmse_db"] for h in history if h["epoch"] == best_epoch),
    )
    if updates != budget["optimizer_updates"]:
        raise RuntimeError("Training did not complete every declared batch")
    write_json_atomic(folder / "training.json", info)
    resume_path.unlink(missing_ok=True)
    return model, info


def fit_polynomial(condition, key, parameters, folder, emit):
    x, y = condition.data["x_train"], condition.data["y_train"]
    start = time.monotonic()
    extra = {}
    if key == "ilc_dpd":
        from opendpd.core.ilc import learn

        xx = to_complex(x[:16384])

        def plant(u):
            iq = np.stack((u.real, u.imag), axis=-1).astype(np.float32)
            return to_complex(pa_output(condition.teacher, iq, condition.device))

        gain_complex = np.vdot(xx, plant(xx)) / np.vdot(xx, xx)
        learned = learn(
            plant,
            xx,
            target_gain=condition.gain,
            inverse_gain=1 / gain_complex,
            iterations=30,
            learning_gain=0.5,
            target_nmse_db=-45,
            peak_limit=condition.peak,
        )
        # Only the training waveform supplies the ILA regression pairs.
        learned_input = learned.input
        post_input = plant(learned_input) / condition.gain
        target = learned_input
        extra = {"ilc_iterations": len(learned.history), "test_feedback": False}
    else:
        # All algorithms identify the same plant, including classical ILA.
        post_input = to_complex(pa_output(condition.teacher, x, condition.device)) / condition.gain
        target = to_complex(x)
    phi = basis(key, parameters, post_input)
    coefficients, diagnostics = fit_least_squares(
        phi, target, float(parameters.get("rcond", 1e-4))
    )
    model = PolynomialModel(key, parameters, coefficients)
    _save_weights(model, folder / "weights.npz")
    info = dict(
        selected_epoch=None,
        attained_epochs=0,
        optimizer_updates=0,
        training_seconds=time.monotonic() - start,
        training_device="cpu",
        fit=diagnostics.to_dict(),
        **extra,
    )
    write_json_atomic(folder / "training.json", info)
    emit("fitting", 0, f"{condition.identifier} · deterministic {key} fit completed")
    return model, info


def cache_binding(base_key, parameters, condition_id, seed):
    return dict(
        training_sha256=arena.training_fingerprint(),
        backbone=base_key,
        model_parameters=parameters,
        condition_id=condition_id,
        seed=seed,
    )


def fit_case(condition, key, parameters, seed, folder, emit, *, ready=None):
    """Train, fit or reload one verified checkpoint. Returns (model, training record)."""
    base_key = get_model(key).weights_from or key
    folder.mkdir(parents=True, exist_ok=True)
    binding = cache_binding(base_key, parameters, condition.identifier, seed)
    if (folder / "training.json").is_file() and (folder / "weights.npz").is_file():
        info = read_json(folder / "training.json")
        if info.get("cache_binding") != binding or info.get(
            "weights_sha256"
        ) != arena.file_hash(folder / "weights.npz"):
            raise ValueError(
                "Cached Arena checkpoint failed request/source integrity verification"
            )
        model = _load_weights(
            build_model(base_key, parameters), folder / "weights.npz"
        ).to(condition.device)
    elif base_key in arena.DETERMINISTIC:
        model, info = fit_polynomial(condition, base_key, parameters, folder, emit)
    else:
        model, info = train_gradient(condition, base_key, parameters, seed, folder, emit, ready=ready)
    info.update(
        cache_binding=binding,
        weights_sha256=arena.file_hash(folder / "weights.npz"),
    )
    write_json_atomic(folder / "training.json", info)
    return model, info


def parameter_count(model):
    return int(
        getattr(model, "n_real_parameters", sum(p.numel() for p in model.parameters()))
    )


def judge_case(condition, model, key):
    """Frozen PA on the independent test input; nothing here selects a checkpoint."""
    x = condition.data["x_test"]
    raw = dpd_output(model, key, x, condition.device)
    if not np.isfinite(raw).all():
        raise FloatingPointError("The candidate produced nonfinite DPD samples")
    amplitude = np.linalg.norm(raw, axis=1)
    u = limit_array(raw, condition.peak)
    baselines = condition.baseline_metrics()
    judges = []
    for name, sha, y in condition.outputs(u):
        metrics = condition.metrics(y, "test")
        baseline = baselines[name]
        judges.append(
            dict(
                judge_id=name,
                checkpoint_sha256=sha,
                **metrics,
                baseline_nmse_db=baseline["nmse_db"],
                baseline_ib_error_db=baseline["ib_error_db"],
                baseline_evm_db=baseline["evm_db"],
                baseline_aclr_l_db=baseline["aclr_l_db"],
                baseline_aclr_r_db=baseline["aclr_r_db"],
                baseline_aer_l_db=baseline["aer_l_db"],
                baseline_aer_r_db=baseline["aer_r_db"],
                baseline_power_error_db=baseline["power_error_db"],
            )
        )
    return dict(
        judges=judges,
        raw_peak_ratio=float(amplitude.max() / condition.peak),
        limited_fraction=float(np.mean(amplitude > condition.peak)),
        teacher_sha256=condition.manifest["teacher"]["sha256"],
        data_sha256=condition.manifest["data_sha256"],
        reference_gain=condition.gain,
        baseline_input_scale_real=float(condition.alpha.real),
        baseline_input_scale_imag=float(condition.alpha.imag),
    )
