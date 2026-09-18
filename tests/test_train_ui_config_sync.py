from src.train_ui import config_sync


def test_load_template_batch_size_reads_train_value(tmp_path):
    """训练页应读取模板中真实的 batch size。"""
    path = tmp_path / "config.json"
    path.write_text('{"train": {"batch_size": 7}}', encoding="utf-8")

    assert config_sync.load_template_batch_size(path, 12) == 7


def test_load_template_batch_size_uses_default_for_invalid_json(tmp_path):
    """模板损坏时读取函数应返回调用方提供的安全默认值。"""
    path = tmp_path / "config.json"
    path.write_text("{", encoding="utf-8")

    assert config_sync.load_template_batch_size(path, 12) == 12


def test_persist_batch_size_rejects_values_below_one(tmp_path, monkeypatch):
    """非法 batch size 不能被写入配置模板。"""
    path = tmp_path / "config.json"
    original = '{"train": {"batch_size": 12}}'
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(config_sync.gr, "update", lambda **kwargs: kwargs)

    update, message = config_sync.persist_batch_size(
        "demo",
        0,
        config_template_path=path,
        default_batch_size=12,
    )

    assert update == {"value": 12}
    assert "必须大于等于 1" in message
    assert path.read_text(encoding="utf-8") == original
