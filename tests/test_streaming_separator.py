import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ExecToolConfig
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class DummyContext:
    def build_messages(self, history, current_message, **kwargs):
        return [{"role": "user", "content": current_message}]

    def add_assistant_message(self, messages, *args, **kwargs):
        return messages

    def add_tool_result(self, messages, *args, **kwargs):
        return messages


class DummySession:
    def __init__(self) -> None:
        self.messages = []

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content})

    def get_history(self, max_messages: int = 50):
        return []


class DummySessionManager:
    def get_or_create(self, session_key: str):
        return DummySession()

    def save(self, session):
        return None


class DummyTool(Tool):
    @property
    def name(self) -> str:
        return "noop"

    @property
    def description(self) -> str:
        return "No-op tool for testing."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "ok"


class StreamingProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def get_default_model(self) -> str:
        return "dummy"

    async def chat(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content="unused")

    async def stream(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield LLMResponse(content="Plan text.")
            yield LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="1", name="noop", arguments={})],
            )
        else:
            yield LLMResponse(content="Result text.")


@pytest.mark.asyncio
async def test_streaming_inserts_separator_between_plan_and_results(tmp_path):
    bus = MessageBus()
    agent = AgentLoop(
        bus=bus,
        provider=StreamingProvider(),
        workspace=tmp_path,
        model="dummy",
        exec_config=ExecToolConfig(),
        session_manager=DummySessionManager(),
    )
    agent.context = DummyContext()
    agent.tools.register(DummyTool())

    chunks: list[str] = []

    def stream_callback(chunk: str) -> None:
        chunks.append(chunk)

    msg = InboundMessage(
        channel="feishu",
        sender_id="user",
        chat_id="oc_1",
        content="hi",
    )

    await agent._process_message(msg, stream_callback=stream_callback)

    assert "".join(chunks) == "Plan text.\n\n---\n\nResult text."
