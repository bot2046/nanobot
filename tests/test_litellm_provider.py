import litellm

from nanobot.providers.litellm_provider import LiteLLMProvider


def test_litellm_provider_disables_aiohttp_transport(monkeypatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False, raising=False)

    LiteLLMProvider()

    assert litellm.disable_aiohttp_transport is True
