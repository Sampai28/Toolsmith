"""Toolsmith MCP server.

Speaks MCP over stdio (default) or HTTP/SSE. The harness drives this process
over a long-lived stdio session rather than calling the handlers in-process,
so the protocol conformance the project claims is the conformance it actually
uses — a server that only works when called directly is not an MCP server.

Guardrails run **here**, on the invocation path, not in the client. A caller
that bypasses the harness entirely still cannot reach `export_respondent_pii`.

The tool surface is selectable per process via `--tool-variant`, which is what
makes the ablation possible: the same server exposes the same capabilities
under four different descriptions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ..data import store
from ..data.store import DatasetError
from ..guardrails.engine import GuardrailEngine, GuardrailLog
from ..tools.variants import VARIANTS, alias_map, render

log = logging.getLogger("toolsmith.mcp")


def build_server(tool_variant: str = "detailed", dataset: Any | None = None) -> Server:
    if tool_variant not in VARIANTS:
        raise ValueError(f"unknown tool variant {tool_variant!r}; expected one of {VARIANTS}")

    rendered = render(tool_variant)
    aliases = alias_map(tool_variant)

    # Loaded once. A server that re-read the corpus per call would make tool
    # latency a disk measurement.
    ds = dataset
    if ds is None:
        try:
            ds = store.default_dataset()
        except DatasetError as exc:
            log.warning("corpus unavailable at startup: %s", exc)
            ds = None

    engine = GuardrailEngine(dataset=ds)
    server: Server = Server("toolsmith")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        # The forbidden tool is listed. Hiding it would make the adversarial
        # suite test discoverability rather than enforcement, and a real
        # deployment cannot assume an attacker does not know a tool exists.
        return [
            types.Tool(
                name=t.name,
                description=t.description,
                inputSchema=t.parameters,
            )
            for t in rendered
        ]

    # validate_input=False hands schema validation to the guardrail layer
    # rather than the SDK.
    #
    # The SDK validates against inputSchema before dispatching, which is the
    # same check argcheck.py performs — but it raises before any guardrail
    # runs, so an out-of-range argument came back as a bare protocol error
    # with no layer, no rule and no log line. The project's claim is that
    # every guardrail decision is attributable; a check that silently
    # pre-empts the attributable one defeats that. Validation is not weakened,
    # it is moved somewhere it can be counted.
    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.ContentBlock]:
        glog = GuardrailLog()
        outcome = engine.invoke(name, arguments or {}, glog, aliases=aliases)

        if not outcome.executed:
            # Returned as an error result rather than raised, so the model sees
            # a refusal it can reason about and the transcript records which
            # rule fired. An exception would surface as a protocol error and
            # tell the caller nothing.
            payload = {
                "error": "blocked_by_guardrail",
                "layer": outcome.blocked_by,
                "rule": outcome.rule,
                "detail": outcome.detail,
                "guardrail_log": glog.to_json(),
            }
            return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

        body = {
            "result": outcome.result,
            "redacted": outcome.redacted,
        }
        return [types.TextContent(type="text", text=json.dumps(body, default=str, indent=2))]

    return server


async def run_stdio(tool_variant: str) -> None:
    server = build_server(tool_variant)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run_sse(tool_variant: str, host: str, port: int) -> None:
    """HTTP/SSE transport, for a client that is not a subprocess."""
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route

    server = build_server(tool_variant)
    transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
        return Response()

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=transport.handle_post_message),
        ]
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Toolsmith MCP server.")
    ap.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    ap.add_argument("--tool-variant", default=os.environ.get("TOOLSMITH_TOOL_VARIANT", "detailed"),
                    choices=list(VARIANTS))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        # stderr: stdout is the MCP transport on this process, and a stray log
        # line there corrupts the protocol stream.
        stream=sys.stderr,
        format="%(levelname)s %(name)s %(message)s",
    )

    if args.transport == "sse":
        run_sse(args.tool_variant, args.host, args.port)
        return 0

    import anyio

    anyio.run(run_stdio, args.tool_variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
