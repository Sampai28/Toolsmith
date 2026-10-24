"""A long-lived MCP stdio session.

The harness executes every tool call through the real MCP server as a
subprocess rather than importing the handlers. That costs a few milliseconds
per call and buys the thing the project claims: the guardrails being measured
are the ones a real MCP client would hit, not a parallel in-process copy that
could drift from them.

The session is opened once per grid cell and reused, because the server's tool
surface is fixed by `--tool-variant` at process start and spawning a process
per call would put interpreter startup inside the latency measurement.

MCP is async; the harness is synchronous. Rather than colouring the whole
runner, the session runs an event loop on a dedicated thread and exposes
blocking methods.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ToolResult:
    ok: bool
    payload: Any
    raw_text: str
    blocked: bool = False
    rule: str | None = None
    layer: str | None = None


class MCPSession:
    """Blocking wrapper around an MCP stdio session on its own event loop."""

    def __init__(self, tool_variant: str = "detailed", python: str | None = None) -> None:
        self.tool_variant = tool_variant
        self._python = python or sys.executable
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._session: ClientSession | None = None
        self._error: BaseException | None = None

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "MCPSession":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self, timeout_s: float = 60.0) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name=f"mcp-{self.tool_variant}")
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise RuntimeError("MCP session did not become ready in time")
        if self._error is not None:
            raise RuntimeError(f"MCP session failed to start: {self._error}")

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:  # recorded, then surfaced by start()
            self._error = exc
            self._ready.set()
        finally:
            try:
                loop.close()
            finally:
                self._loop = None

    async def _serve(self) -> None:
        params = StdioServerParameters(
            command=self._python,
            args=["-m", "toolsmith.server.mcp_server",
                  "--transport", "stdio",
                  "--tool-variant", self.tool_variant],
            env=None,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                # Hold the context open until close() is called. The streams
                # and the subprocess are owned by these context managers, so
                # the coroutine has to stay alive for the session to.
                while not self._stop.is_set():
                    await asyncio.sleep(0.05)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15.0)
            self._thread = None
        self._session = None

    def _submit(self, coro) -> Any:
        loop = self._loop
        if loop is None or self._session is None:
            raise RuntimeError("MCP session is not running")
        fut: Future = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout=120.0)

    # -- protocol --------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        async def _go():
            assert self._session is not None
            res = await self._session.list_tools()
            return [
                {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
                for t in res.tools
            ]

        return self._submit(_go())

    def call_tool(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        async def _go():
            assert self._session is not None
            return await self._session.call_tool(name, arguments or {})

        try:
            res = self._submit(_go())
        except Exception as exc:
            return ToolResult(ok=False, payload=None, raw_text=f"{type(exc).__name__}: {exc}")

        text = ""
        for block in getattr(res, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text += getattr(block, "text", "")

        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            return ToolResult(ok=False, payload=None, raw_text=text)

        if isinstance(payload, dict) and payload.get("error") == "blocked_by_guardrail":
            return ToolResult(
                ok=False,
                payload=payload,
                raw_text=text,
                blocked=True,
                rule=payload.get("rule"),
                layer=payload.get("layer"),
            )

        result = payload.get("result") if isinstance(payload, dict) else payload
        return ToolResult(ok=True, payload=result, raw_text=text)
