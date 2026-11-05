"""Channel defaults affect idle displays without changing turn or switch routing."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, build_session_key
from gateway.slash_commands_status import _status_model_route
from tests.gateway.test_status_command import _make_runner, _make_source, _make_event


@pytest.fixture
def channel_runner(tmp_path, monkeypatch):
    source = _make_source()
    entry = SessionEntry(session_key=build_session_key(source), session_id="new-session",
                         created_at=datetime.now(), updated_at=datetime.now(), platform=Platform.TELEGRAM)
    entry.last_prompt_tokens = 100
    runner = _make_runner(entry)
    runner.config.platforms[Platform.TELEGRAM].channel_overrides = {
        "c1": ChannelOverride(provider="anthropic", model="channel-model")}
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("model:\n  default: global-model\n  provider: openrouter\n")
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"provider": "openrouter"})
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", lambda *a, **kw: {
        "provider": "anthropic", "base_url": "", "api_key": "channel-key"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length",
                        lambda *a, **kw: 64000 if kw.get("provider") == "anthropic" else 32000)
    return runner, entry, source


@pytest.fixture(params=[False, True], ids=["channel", "unavailable"])
def display_case(channel_runner, request, monkeypatch):
    if request.param:
        def unavailable(*a, **kw):
            raise RuntimeError("unavailable")
        monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", unavailable)
        return "global-model", "openrouter", "channel-model", 32000
    return "channel-model", "anthropic", "global-model", 64000


@pytest.mark.asyncio
async def test_status_channel_display(channel_runner, display_case):
    runner, _, _ = channel_runner
    model, provider, rejected, _ = display_case
    reply = await runner._handle_status_command(_make_event("/status"))
    assert model in reply
    assert provider in reply.lower()
    assert rejected not in reply


@pytest.mark.asyncio
async def test_context_channel_window(channel_runner, display_case):
    runner, entry, source = channel_runner
    model, _, rejected, window = display_case
    reply = await runner._handle_context_command(_make_event("/context"))
    figures = await runner._resolve_context_figures(None, None, entry, source)
    assert figures[1:3] == (window, model)
    assert model in reply
    assert rejected not in reply


@pytest.mark.asyncio
async def test_footer_channel_preview(channel_runner, display_case, tmp_path, monkeypatch):
    runner, _, _ = channel_runner
    model, _, rejected, _ = display_case
    monkeypatch.setattr(runner, "_display_config_target", lambda *a: (tmp_path / "config.yaml", "telegram"))
    event = _make_event("/footer on")
    event.message = "/footer on"
    reply = await runner._handle_footer_command(event)
    assert model in reply
    assert rejected not in reply


@pytest.mark.asyncio
async def test_bare_model_channel_current(channel_runner, display_case, tmp_path, monkeypatch):
    from tests.gateway.test_model_command_context_offload import _runner_with_store, _event
    runner = _runner_with_store(tmp_path, monkeypatch)
    runner.config = channel_runner[0].config
    model, provider, rejected, _ = display_case
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **kw: 64000)
    (tmp_path / ".hermes" / "config.yaml").write_text(
        "model:\n  default: global-model\n  provider: openrouter\n")
    reply = await runner._handle_model_command(_event("/model"))
    assert model in reply
    assert provider in reply.lower()
    assert rejected not in reply


@pytest.mark.parametrize("live", [False, True])
def test_status_served_route_beats_channel(live):
    agent = SimpleNamespace(model="live-model", provider="live-provider") if live else None
    result = _status_model_route(
        agent, {}, {"model": "served-model", "billing_provider": "served-provider"},
        {"model": "row-model", "billing_provider": "row-provider"}, None,
        channel_route={"model": "channel-model", "provider": "channel-provider"},
    )
    assert result[:2] == (("live-model", "live-provider") if live else ("served-model", "served-provider"))


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True])
async def test_context_known_model_beats_channel(channel_runner, live):
    runner, entry, source = channel_runner
    runner._session_db = SimpleNamespace(get_session=AsyncMock(return_value={"model": "row-model"}))
    agent = SimpleNamespace(model="live-model", provider="live-provider") if live else None
    runner._resolve_route_context = AsyncMock(return_value=SimpleNamespace(context_length=64000, context_source="detected"))
    figures = await runner._resolve_context_figures(agent, None, entry, source)
    assert figures[2] == ("live-model" if live else "row-model")


@pytest.mark.asyncio
async def test_typed_model_switch_ignores_channel(tmp_path, monkeypatch):
    from tests.gateway.test_model_command_context_offload import _runner_with_store, _event
    from hermes_cli.model_switch import ModelSwitchResult
    runner = _runner_with_store(tmp_path, monkeypatch)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(channel_overrides={
        "c1": ChannelOverride(provider="anthropic", model="channel-model")})})
    seen = {}
    def switch(**kw):
        seen.update(kw)
        return ModelSwitchResult(success=True, new_model="new-model", target_provider="openrouter",
                                 api_key="test", base_url="https://openrouter.ai/api/v1")
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", switch)
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **kw: 64000)
    runner._resolve_display_route = AsyncMock(side_effect=AssertionError("typed switch must not resolve display route"))
    reply = await runner._handle_model_command(_event("/model new-model"))
    assert "new-model" in reply
    assert seen["current_provider"] == "openrouter"
    assert seen["current_model"] == "old-model"
