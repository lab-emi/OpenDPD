"""The public package and its result evidence contain only the released dataset."""
import io
import json
import shutil
import sys
import zipfile

import pytest

from opendpd.core import arena
from opendpd.web.gpu_archive import pack


def test_apa_b_is_the_only_published_dataset_and_all_rows_are_complete():
    protocol = arena.protocol()
    assert [(b.board_id, b.dataset) for b in protocol.boards] == [("apa-200mhz-b", "APA_200MHz_b")]
    assert set(arena.calibration()) == set(protocol.pa_models) == {"apa-200mhz-b"}
    rows = arena.load_official_rows()
    assert len(rows) == 23 and sum(r.completed_cases for r in rows) == 233
    assert all(r.board_id == "apa-200mhz-b" and r.status == "succeeded" for r in rows)
    assert all(c["condition_id"] == "apa-200mhz-b" and c["evaluation_split"] == "test"
               for r in rows for c in r.cases)
    assert all(r.provenance["source_training_sha256"] != protocol.training_sha256 for r in rows)
    for row in rows:
        for case in row.cases:
            for reviewed in (case.get("execution_profile") or {}).get("replay_units", []):
                assert reviewed["unit"][2] == "apa-200mhz-b"
    for excluded in ("dpa-160mhz", "dpa-200mhz", "apa-200mhz", "synthetic-suite"):
        with pytest.raises(ValueError, match="Unknown Arena leaderboard"):
            arena.board(excluded)


def test_selected_arena_outputs_use_the_opened_run_subtree(tmp_path):
    run = tmp_path / "runs" / "run-test"
    (run / "logs").mkdir(parents=True)
    (run / "result.json").write_text('{"status":"succeeded"}')
    (run / "logs/worker.log").write_text("completed")
    (run / "not-exported.txt").write_text("private training state")
    data = pack(tmp_path, [run / "result.json", run / "logs/worker.log"], subtree="runs/run-test")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert set(archive.namelist()) == {"result.json", "logs/worker.log"}
        assert archive.read("result.json") == b'{"status":"succeeded"}'
    with pytest.raises(ValueError):
        pack(tmp_path, [tmp_path / "outside.json"], subtree="runs/run-test")


@pytest.mark.parametrize("archived_prior", [False, True])
def test_documented_pa_refit_loads_packaged_calibration_and_limits_release_scope(
    tmp_path, monkeypatch, archived_prior
):
    from benchmark import retrain_arena_pa

    expected = arena.calibration()
    args = ["retrain_arena_pa", "--workspace", str(tmp_path / "pa"), "--device", "cpu"]
    if archived_prior:
        prior = tmp_path / "archived-calibration.json"
        prior.write_text(json.dumps({**expected, "dpa-160mhz": {"unpublished": True}}))
        args += ["--prior-calibration", str(prior)]
    captured = []
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(retrain_arena_pa.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(retrain_arena_pa, "prepare", lambda *values: captured.append(values))
    retrain_arena_pa.main()
    assert captured == [(tmp_path / "pa", expected, "cpu")]


def test_pa_refit_can_retain_the_already_packaged_validation_winner(tmp_path, monkeypatch):
    from benchmark import retrain_arena_pa as refit

    prior = arena.calibration()
    meta = prior["apa-200mhz-b"]
    assets = tmp_path / "assets"
    assets.mkdir()
    for filename in (meta["data_file"], meta["teacher"]["file"]):
        shutil.copyfile(arena.ASSETS / filename, assets / filename)
    original_hash = arena.file_hash(assets / meta["teacher"]["file"])
    monkeypatch.setattr(arena, "ASSETS", assets)

    class Model:
        def to(self, _):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(refit.e, "load_frozen", lambda *_: Model())
    monkeypatch.setattr(refit.e, "build_model", lambda *_: Model())
    monkeypatch.setattr(refit.e, "_load_weights", lambda model, _: model)
    monkeypatch.setattr(refit.e, "pa_output", lambda _model, x, _device: x)
    monkeypatch.setattr(refit, "validation_metrics", lambda *_: {"nmse_db": -50.0})
    monkeypatch.setattr(refit, "adjacent_error_ratios", lambda *_: {})
    monkeypatch.setattr(refit, "train", lambda *_: {
        "validation_nmse_db": -40.0, "parameters": 4871, "sha256": "unused-candidate"
    })
    refit.prepare(tmp_path, prior, "cpu")
    selected = json.loads((tmp_path / "pa-selected.json").read_text())["apa-200mhz-b"]
    assert selected["teacher"]["source"] == "retained PA on identical measured partitions"
    assert selected["teacher"]["sha256"] == original_hash
    assert arena.file_hash(assets / meta["teacher"]["file"]) == original_hash
