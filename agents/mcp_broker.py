"""
mcp_broker.py — Model Context Protocol message broker server.

Serves as the central communication hub between Claude (principal engineer)
and Gemini (orchestrator). Agents register, post artifacts, and poll for
directives through this server.

Run:
    python agents/mcp_broker.py

Default port: 7823
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock, Thread
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mcp_broker")

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
_lock = Lock()

# Registered agents: { agent_id: { name, role, registered_at, last_seen } }
_agents: dict[str, dict] = {}

# Message queues per agent: { agent_id: [ message, ... ] }
_queues: dict[str, list] = defaultdict(list)

# Artifact store: { artifact_id: { from, to, type, payload, created_at } }
_artifacts: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# MCP JSON-RPC helpers
# ---------------------------------------------------------------------------
JSONRPC_VERSION = "2.0"


def ok(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result}


def err(req_id: Any, code: int, message: str) -> dict:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": req_id,
        "error": {"code": code, "message": message},
    }


# ---------------------------------------------------------------------------
# Tool handlers — these implement the broker's MCP tool surface
# ---------------------------------------------------------------------------

def handle_register(params: dict, req_id: Any) -> dict:
    """
    Register an agent with the broker.

    params: { agent_id?, name, role }
    Returns: { agent_id, status }
    """
    name = params.get("name", "").strip()
    role = params.get("role", "").strip()
    if not name or not role:
        return err(req_id, -32602, "'name' and 'role' are required")

    agent_id = params.get("agent_id") or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with _lock:
        _agents[agent_id] = {
            "name": name,
            "role": role,
            "registered_at": now,
            "last_seen": now,
        }
        if agent_id not in _queues:
            _queues[agent_id] = []

    log.info("REGISTER  agent_id=%s  name=%s  role=%s", agent_id, name, role)
    return ok(req_id, {"agent_id": agent_id, "status": "registered"})


def handle_list_agents(params: dict, req_id: Any) -> dict:
    """Return all registered agents."""
    with _lock:
        snapshot = {aid: dict(info) for aid, info in _agents.items()}
    return ok(req_id, {"agents": snapshot})


def handle_send(params: dict, req_id: Any) -> dict:
    """
    Send a message from one agent to another.

    params: { from_agent, to_agent, content, message_type? }
    Returns: { message_id, queued_at }
    """
    from_agent = params.get("from_agent", "").strip()
    to_agent = params.get("to_agent", "").strip()
    content = params.get("content")
    if not from_agent or not to_agent or content is None:
        return err(req_id, -32602, "'from_agent', 'to_agent', and 'content' are required")

    with _lock:
        if to_agent not in _agents:
            return err(req_id, -32001, f"Unknown recipient agent: {to_agent}")

        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        message = {
            "message_id": msg_id,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": params.get("message_type", "directive"),
            "content": content,
            "sent_at": now,
            "read": False,
        }
        _queues[to_agent].append(message)

        # Update sender last_seen
        if from_agent in _agents:
            _agents[from_agent]["last_seen"] = now

    log.info("SEND  %s → %s  msg_id=%s", from_agent, to_agent, msg_id)
    return ok(req_id, {"message_id": msg_id, "queued_at": now})


def handle_poll(params: dict, req_id: Any) -> dict:
    """
    Poll for pending messages for an agent. Marks fetched messages as read.

    params: { agent_id, mark_read? (default true) }
    Returns: { messages: [...] }
    """
    agent_id = params.get("agent_id", "").strip()
    if not agent_id:
        return err(req_id, -32602, "'agent_id' is required")

    mark_read = params.get("mark_read", True)

    with _lock:
        if agent_id not in _agents:
            return err(req_id, -32001, f"Unknown agent: {agent_id}")

        pending = [m for m in _queues[agent_id] if not m["read"]]
        if mark_read:
            for m in _queues[agent_id]:
                m["read"] = True

        _agents[agent_id]["last_seen"] = datetime.now(timezone.utc).isoformat()

    log.info("POLL  agent_id=%s  pending=%d", agent_id, len(pending))
    return ok(req_id, {"messages": pending})


def handle_post_artifact(params: dict, req_id: Any) -> dict:
    """
    Store a named artifact (e.g. an implementation plan or backtest result).

    params: { from_agent, to_agent, artifact_type, payload }
    Returns: { artifact_id }
    """
    from_agent = params.get("from_agent", "").strip()
    to_agent = params.get("to_agent", "").strip()
    artifact_type = params.get("artifact_type", "document").strip()
    payload = params.get("payload")

    if not from_agent or not to_agent or payload is None:
        return err(req_id, -32602, "'from_agent', 'to_agent', and 'payload' are required")

    artifact_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with _lock:
        _artifacts[artifact_id] = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "artifact_type": artifact_type,
            "payload": payload,
            "created_at": now,
        }

        # Auto-notify recipient via their message queue
        if to_agent in _agents:
            notification = {
                "message_id": str(uuid.uuid4()),
                "from_agent": from_agent,
                "to_agent": to_agent,
                "message_type": "artifact_notification",
                "content": {
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "note": f"New artifact posted by {from_agent}",
                },
                "sent_at": now,
                "read": False,
            }
            _queues[to_agent].append(notification)

    log.info("ARTIFACT  %s → %s  id=%s  type=%s", from_agent, to_agent, artifact_id, artifact_type)
    return ok(req_id, {"artifact_id": artifact_id, "created_at": now})


def handle_get_artifact(params: dict, req_id: Any) -> dict:
    """
    Retrieve a stored artifact by ID.

    params: { artifact_id }
    Returns: { artifact }
    """
    artifact_id = params.get("artifact_id", "").strip()
    if not artifact_id:
        return err(req_id, -32602, "'artifact_id' is required")

    with _lock:
        artifact = _artifacts.get(artifact_id)

    if artifact is None:
        return err(req_id, -32001, f"Artifact not found: {artifact_id}")

    return ok(req_id, {"artifact": artifact})


def handle_status(params: dict, req_id: Any) -> dict:
    """Return broker health / stats."""
    with _lock:
        total_pending = sum(
            sum(1 for m in msgs if not m["read"]) for msgs in _queues.values()
        )
        stats = {
            "status": "ok",
            "registered_agents": len(_agents),
            "total_artifacts": len(_artifacts),
            "total_pending_messages": total_pending,
            "broker_time": datetime.now(timezone.utc).isoformat(),
        }
    return ok(req_id, stats)


# ---------------------------------------------------------------------------
# Tool dispatch table — maps MCP method names → handlers
# ---------------------------------------------------------------------------
TOOLS = {
    "broker/register": handle_register,
    "broker/list_agents": handle_list_agents,
    "broker/send": handle_send,
    "broker/poll": handle_poll,
    "broker/post_artifact": handle_post_artifact,
    "broker/get_artifact": handle_get_artifact,
    "broker/status": handle_status,
}

# ---------------------------------------------------------------------------
# HTTP request handler (JSON-RPC 2.0 over HTTP POST)
# ---------------------------------------------------------------------------

class MCPHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default HTTP logs; we use our own
        pass

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/"):
            with _lock:
                body = {
                    "service": "mcp_broker",
                    "status": "ok",
                    "agents": list(_agents.keys()),
                    "uptime_since": _start_time,
                }
            self._send_json(body)
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        try:
            rpc = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(err(None, -32700, f"Parse error: {exc}"), 400)
            return

        req_id = rpc.get("id")
        method = rpc.get("method", "")
        params = rpc.get("params", {})

        handler = TOOLS.get(method)
        if handler is None:
            self._send_json(err(req_id, -32601, f"Method not found: {method}"), 404)
            return

        try:
            result = handler(params, req_id)
        except Exception as exc:
            log.exception("Unhandled error in handler %s", method)
            result = err(req_id, -32603, f"Internal error: {exc}")

        self._send_json(result)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 7823
_start_time = datetime.now(timezone.utc).isoformat()


def main():
    server = HTTPServer((HOST, PORT), MCPHandler)
    log.info("MCP Broker running on http://%s:%d", HOST, PORT)
    log.info("Available methods: %s", list(TOOLS.keys()))
    log.info("Health check: GET http://%s:%d/health", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Broker shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
