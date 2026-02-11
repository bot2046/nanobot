from nanobot.channels.feishu import FeishuChannel
from pathlib import Path


def test_extract_explicit_attachments() -> None:
    text = "hello ![](/tmp/a.png) file:/tmp/b.txt world"
    cleaned, paths = FeishuChannel._extract_explicit_attachments(text)
    assert "/tmp/a.png" in paths
    assert "/tmp/b.txt" in paths
    assert "hello" in cleaned and "world" in cleaned


def test_normalize_attachment_paths(tmp_path: Path) -> None:
    ok = tmp_path / "ok.png"
    ok.write_bytes(b"x")
    cleaned = FeishuChannel._normalize_attachment_paths([str(ok), "rel.png", "/nope.jpg"])
    assert cleaned == [ok]
