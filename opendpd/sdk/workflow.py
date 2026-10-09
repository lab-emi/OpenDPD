"""One call from paired I/Q to a trained PA and DPD, written as ``opendpd-model-v1`` packages.

This is the SDK function behind the MATLAB toolbox's ``opendpd.fit``. It does what a person does in the project API,
in order: import the capture (nothing is normalised), train a PA model, train a DPD through that PA, wait for both and
export both. Everything is an ordinary run in the workspace, so it is reproducible and can be opened in Studio. The
models must be exportable (``opendpd.services.model_export.EXPORT_MODELS``) and the PA model must be gradient-trained
(the DPD is trained through it): both are checked before anything is trained, because a refusal after an hour of
training would be the worst time to find out.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .client import SDKError, TERMINAL, open_project

CANCEL_GRACE_SECONDS = 60.0


def _event(emit: Optional[Callable[[Dict[str, Any]], None]], **fields):
    if emit is not None:
        emit(fields)


def _wait(job, stage: str, *, deadline: Optional[float], cancelled: Optional[Callable[[], bool]],
          emit: Optional[Callable[[Dict[str, Any]], None]], poll_interval: float):
    """Wait for ``job``; cancel it (and raise) on request or timeout. Reports (status, epoch) changes as events."""
    seen = None
    cancelling = None
    while True:
        record = job.status()
        marker = (record["status"], record.get("progress_epoch"))
        if marker != seen:
            seen = marker
            _event(emit, stage=stage, run_id=job.run_id, status=record["status"], epoch=record.get("progress_epoch"),
                   total_epochs=record.get("progress_total_epochs"))
        if record["status"] in TERMINAL:
            if record["status"] == "succeeded":
                return record
            error = record.get("error") or {}
            if cancelling:
                raise SDKError(cancelling, f"Run {job.run_id} was cancelled ({stage})")
            raise SDKError(record["status"], f"Run {job.run_id} ({stage}) ended as {record['status']}: "
                                             f"{error.get('message', record.get('status_reason') or record['status'])}")
        now = time.monotonic()
        if cancelling is None:
            if cancelled is not None and cancelled():
                cancelling = "cancelled"
            elif deadline is not None and now >= deadline:
                cancelling = "timeout"
            if cancelling:
                job.cancel()
                cancel_deadline = now + CANCEL_GRACE_SECONDS
        elif now >= cancel_deadline:
            raise SDKError(cancelling, f"Run {job.run_id} did not stop within {CANCEL_GRACE_SECONDS:g} s of the "
                                       "cancel request; it is still running in the workspace")
        time.sleep(poll_interval)


def _settle(job, seconds: float = 30.0):
    """Give a job that was just asked to cancel a moment to reach a terminal state."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            if job.status()["status"] in TERMINAL:
                return
        except SDKError:
            return
        time.sleep(0.2)


def _release(project, started: bool, emit: Optional[Callable[[Dict[str, Any]], None]]):
    """Disconnect from the workspace service. It is stopped only if this call started it and no other run is using it;
    a service that was already there (another session's, or a Studio the person has open) is left alone, and so is one
    that still has work. A failure here is reported as an event and never raised: it must not hide the result of a fit
    that worked or replace the error of one that did not."""
    try:
        project.close(stop_service=started and project.active_run_count() == 0)
    except SDKError as err:
        _event(emit, stage="close", warning=str(err))


def fit(workspace, x, y, *, sample_rate_hz, bandwidth_hz, nperseg, dpd_model="gru", pa_model="gru",
        dpd_parameters=None, pa_parameters=None, training=None, device="auto", device_index=0, num_threads=None,
        profile="opendpd-spectral-v2", dataset_id=None, display_name=None, origin="unknown", amplitude_units="unknown",
        guard_samples=256, n_sub_ch=1, output_folder=None, timeout=None,
        on_event=None, cancelled=None, poll_interval=0.5) -> Dict[str, Any]:
    """Import ``x`` (PA input) and ``y`` (PA output), train a PA and a DPD, export both; returns a JSON-able summary.

    ``nperseg`` has no default (see ``Project.import_iq``). ``on_event`` receives dicts with ``stage``
    (``import``, ``train_pa``, ``train_dpd``, ``export``) and, while training, ``run_id``, ``status``, ``epoch`` and
    ``total_epochs``. ``cancelled`` is polled; when it returns true the running job is cancelled and ``SDKError``
    ``cancelled`` is raised. ``timeout`` (seconds, ``None`` for none) works the same way for the whole training and
    raises ``timeout``. Packages go to ``output_folder`` (default ``<workspace>/exports``) as ``<run_id>.opendpd.zip``.
    """
    from opendpd.services.model_export import EXPORT_MODELS

    for role, key in (("dpd_model", dpd_model), ("pa_model", pa_model)):
        if key not in EXPORT_MODELS:
            raise ValueError(f"{role} '{key}' cannot be exported; fit supports {', '.join(EXPORT_MODELS)}")
    from opendpd.core.registry import get_model

    if get_model(pa_model).training_method != "gradient":
        raise ValueError(f"pa_model '{pa_model}' is a least-squares baseline, not a DPD surrogate: the DPD is trained through "
                         "the PA model, so pa_model must be gradient-trained (gru, tres_gru or gmp)")
    if timeout is not None and not timeout > 0:
        raise ValueError("timeout must be positive")
    if not poll_interval > 0:
        raise ValueError("poll_interval must be positive")
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    started = time.monotonic()
    project = open_project(workspace)
    started = project.started_service
    running = None
    try:
        folder = Path(output_folder).expanduser().resolve() if output_folder else project.workspace / "exports"
        _event(on_event, stage="import")
        dataset = project.import_iq(x, y, dataset_id=dataset_id, display_name=display_name, sample_rate_hz=sample_rate_hz,
                                    bandwidth_hz=bandwidth_hz, nperseg=nperseg, n_sub_ch=n_sub_ch,
                                    guard_samples=guard_samples, origin=origin, amplitude_units=amplitude_units)
        wait = dict(deadline=deadline, cancelled=cancelled, emit=on_event, poll_interval=poll_interval)
        pa = running = project.train_pa(dataset["dataset_id"], model=pa_model, parameters=pa_parameters, training=training,
                                        device=device, device_index=device_index, num_threads=num_threads, profile=profile)
        _wait(pa, "train_pa", **wait)
        dpd = running = project.train_dpd(dataset["dataset_id"], pa, model=dpd_model, parameters=dpd_parameters,
                                          training=training, device=device, device_index=device_index,
                                          num_threads=num_threads, profile=profile)
        _wait(dpd, "train_dpd", **wait)
        running = None
        packages = {}
        for role, job in (("pa", pa), ("dpd", dpd)):
            _event(on_event, stage="export", role=role, run_id=job.run_id)
            packages[role] = job.export(folder / f"{job.run_id}.opendpd.zip")
        return {"workspace": str(project.workspace), "dataset": dataset, "seconds": time.monotonic() - started,
                "pa": {"run_id": pa.run_id, "status": "succeeded", "result": pa.result(), "package": packages["pa"]},
                "dpd": {"run_id": dpd.run_id, "status": "succeeded", "result": dpd.result(), "package": packages["dpd"]}}
    except BaseException:
        if running is not None:                # never leave a run training after the caller has given up
            try:
                if running.status()["status"] not in TERMINAL:
                    running.cancel()
                    _settle(running)
            except Exception:
                pass
        raise
    finally:
        _release(project, started, on_event)
