from nanobot.channels.feishu import FeishuChannel


def test_extract_explicit_attachments() -> None:
    text = "hello ![](/tmp/a.png) file:/tmp/b.txt world"
    cleaned, paths = FeishuChannel._extract_explicit_attachments(text)
    assert "/tmp/a.png" in paths
    assert "/tmp/b.txt" in paths
    assert "hello" in cleaned and "world" in cleaned
