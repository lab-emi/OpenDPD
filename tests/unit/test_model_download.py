from opendpd.services.model_download import MODEL_FILE, MODEL_META, publish_model, read_model


def test_snapshot_never_serves_mixed_or_partial_checkpoints(tmp_path):
    assert read_model(tmp_path) is None
    publish_model(tmp_path, b'first complete model', epoch=1)
    previous = read_model(tmp_path)
    (tmp_path / MODEL_FILE).write_bytes(b'incomplete replacement')
    assert read_model(tmp_path) is None
    publish_model(tmp_path, b'second complete model', epoch=2)
    current = read_model(tmp_path)
    assert previous[1] == b'first complete model'
    assert current[0]['epoch'] == 2 and current[1] == b'second complete model'
    assert current[0]['sha256'] != previous[0]['sha256']
    assert not (tmp_path / '.model-download.tmp').exists()
    (tmp_path / MODEL_META).unlink()
    (tmp_path / MODEL_META).symlink_to(tmp_path / MODEL_FILE)
    assert read_model(tmp_path) is None
