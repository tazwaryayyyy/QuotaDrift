import json
import uuid
import asyncio
from fastapi import Request
from fastapi.responses import StreamingResponse

# This is a minimal, low-dependency implementation of the MCP SSE transport.
# It allows QuotaDrift to act as an MCP server for Claude Code, Cursor, etc.

class MCPServer:
    def __init__(self):
        self.clients = {} # session_id -> queue

    async def sse_handler(self, request: Request):
        client_id = str(uuid.uuid4())
        queue = asyncio.Queue()
        self.clients[client_id] = queue

        async def _gen():
            # 1. Initial 'endpoint' event tells the client where to POST messages
            # In a real setup, this would be the absolute URL to /mcp/messages
            # For QuotaDrift, we'll assume relative or handle it in main.py
            yield f"event: endpoint\ndata: /mcp/messages?client_id={client_id}\n\n"
            
            try:
                while True:
                    msg = await queue.get()
                    yield f"data: {json.dumps(msg)}\n\n"
            except asyncio.CancelledError:
                del self.clients[client_id]

        return StreamingResponse(_gen(), media_type="text/event-stream")

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
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "QuotaDrift",
                        "version": "1.0.0"
                    }
                }
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
                                    "project_id": {"type": "integer"}
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "read_file",
                            "description": "Read the content of a specific file from the project.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "filename": {"type": "string"},
                                    "project_id": {"type": "integer"}
                                },
                                "required": ["filename"]
                            }
                        }
                    ]
                }
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
                        "result": {
                            "content": [{"type": "text", "text": str(result)}]
                        }
                    }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32000, "message": str(e)}
                    }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": "Method not found"}
        }

mcp = MCPServer()
