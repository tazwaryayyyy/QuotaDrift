import asyncio
import json
import logging
import time
import uuid

from fastapi import Request
from fastapi.responses import StreamingResponse

# This is a minimal, low-dependency implementation of the MCP SSE transport.
# It allows QuotaDrift to act as an MCP server for Claude Code, Cursor, etc.

logger = logging.getLogger("mcp_server")


class MCPServer:
    def __init__(self):
        # session_id -> {"queue": asyncio.Queue, "last_seen": float}
        self.clients: dict[str, dict] = {}

    async def sse_handler(self, _request: Request):
        client_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        self.clients[client_id] = {"queue": queue, "last_seen": time.time()}

        async def _gen():
            yield f"event: endpoint\ndata: /mcp/messages?client_id={client_id}\n\n"
            try:
                while True:
                    msg = await self.clients[client_id]["queue"].get()
                    self.clients[client_id]["last_seen"] = time.time()
                    yield f"data: {json.dumps(msg)}\n\n"
            finally:
                # Guarantee cleanup on ANY exit path: normal close, cancel,
                # or exception. The original code only ran on CancelledError,
                # leaking the entry on TCP disconnects and HTTP/2 resets.
                self.clients.pop(client_id, None)

        return StreamingResponse(_gen(), media_type="text/event-stream")

    async def reap_stale_clients(self, max_idle_seconds: float = 300.0) -> None:
        """Background task that evicts clients that have been silent for too long.

        The SSE generator's finally-block handles cleanup for most disconnects,
        but some proxies and load balancers hold TCP connections open without
        forwarding traffic. This reaper provides a backstop so those zombie
        entries don't accumulate unboundedly.

        Run with: asyncio.create_task(mcp.reap_stale_clients())
        """
        while True:
            await asyncio.sleep(60)
            now = time.time()
            stale = [
                cid
                for cid, info in list(self.clients.items())
                if now - info["last_seen"] > max_idle_seconds
            ]
            for cid in stale:
                self.clients.pop(cid, None)
                logger.info(
                    "Reaped stale MCP client %s (idle > %.0fs)", cid, max_idle_seconds)

    async def handle_message(self, client_id: str, message: dict, tools_registry: dict):
        if client_id not in self.clients:
            return {"error": "Invalid client_id"}

        # Basic JSON-RPC handling
        msg_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "QuotaDrift", "version": "1.0.0"},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        {
                            "name": "search_codebase",
                            "description": "Search the indexed project files using hybrid RAG (Vector + BM25).",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "project_id": {"type": "integer"},
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "read_file",
                            "description": "Read the content of a specific file from the project.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "filename": {"type": "string"},
                                    "project_id": {"type": "integer"},
                                },
                                "required": ["filename"],
                            },
                        },
                    ]
                },
            }

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            if tool_name in tools_registry:
                try:
                    result = await tools_registry[tool_name](**tool_args)
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"content": [{"type": "text", "text": str(result)}]},
                    }
                except (TypeError, ValueError, RuntimeError, KeyError) as exc:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32000, "message": str(exc)},
                    }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": "Method not found"},
        }


mcp = MCPServer()
