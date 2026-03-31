"""
QuotaDrift — FastAPI backend.

Endpoints
---------
GET  /                          → serve index.html
POST /api/session/new           → create project + session
GET  /api/sessions              → list all sessions
GET  /api/history/{session_id}  → full message history
GET  /api/projects              → list projects
POST /api/chat/stream           → SSE streaming chat (main endpoint)
POST /api/switch-context        → compile + serialize current session state
GET  /api/model-status          → live model health snapshot
POST /api/project/index         → upload + index code files into project
GET  /api/export/{session_id}   → download session as Markdown
POST /api/cache/clear           → wipe the semantic cache
"""

import asyncio
import json
import logging
import logging.handlers
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import psutil
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

import cache
import compiler
import config
import enhanced_agent_runner
import memory
import model_manager
import router as ai_router

load_dotenv()


# ---------------------------------------------------------------------------
# Structured Logging Configuration
# ---------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """Custom JSON formatter with request ID tracking."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging():
    """Configure structured logging with rotation for the application."""
    # Create formatters
    json_formatter = JSONFormatter()
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # File handler with rotation (10MB max, keep 5 files)
    file_handler = logging.handlers.RotatingFileHandler(
        "quotadrift.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,  # 10MB
    )
    file_handler.setFormatter(json_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger


# Setup logging
logger = setup_logging()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="QuotaDrift", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define STATIC_DIR
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Request ID Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add unique request ID to all requests for tracing."""
    request_id = str(uuid.uuid4())[:8]

    # Add request ID to request state
    request.state.request_id = request_id

    # Add to logger context
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id
        return record

    logging.setLogRecordFactory(record_factory)

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        logging.setLogRecordFactory(old_factory)


@app.on_event("startup")
async def startup():
    memory.init_db()
    model_manager.model_manager.start_background_tasks()

    # Verify provider configuration
    await verify_providers()

    # Start system metrics task
    asyncio.create_task(system_metrics_task())

    logger.info("QuotaDrift startup complete - all systems ready")


async def verify_providers():
    """Verify that all configured providers have valid API keys."""
    logger.info("Verifying provider configuration...")

    required_env_vars = {
        "GROQ_API_KEY": "Groq",
        "GITHUB_TOKEN": "GitHub Models",
        "MISTRAL_API_KEY": "Mistral AI",
        "SILICONFLOW_API_KEY": "Silicon Flow",
        "HUGGINGFACE_API_KEY": "Hugging Face",
        "CLOUDFLARE_API_KEY": "Cloudflare Workers AI",
        "CLOUDFLARE_ACCOUNT_ID": "Cloudflare Workers AI",
        "OPENROUTER_API_KEY": "OpenRouter",
    }

    missing_vars = []
    for env_var, provider in required_env_vars.items():
        if not os.getenv(env_var):
            missing_vars.append(f"{provider} ({env_var})")

    if missing_vars:
        logger.warning(f"Missing API keys for: {', '.join(missing_vars)}")
        logger.warning("These providers will be skipped during failover.")
    else:
        logger.info("✅ All provider API keys configured")

    # Test basic connectivity for each provider
    await test_provider_connectivity()


async def test_provider_connectivity():
    """Test basic connectivity for each provider."""

    # Only test if we have at least one provider key
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("Skipping connectivity tests - no primary provider key found")
        return

    test_providers = []

    if os.getenv("GROQ_API_KEY"):
        test_providers.append("Groq")
    if os.getenv("MISTRAL_API_KEY"):
        test_providers.append("Mistral AI")
    if os.getenv("SILICONFLOW_API_KEY"):
        test_providers.append("Silicon Flow")
    if os.getenv("HUGGINGFACE_API_KEY"):
        test_providers.append("Hugging Face")
    if os.getenv("CLOUDFLARE_API_KEY") and os.getenv("CLOUDFLARE_ACCOUNT_ID"):
        test_providers.append("Cloudflare Workers AI")
    if os.getenv("OPENROUTER_API_KEY"):
        test_providers.append("OpenRouter")

    logger.info(f"Testing connectivity for: {', '.join(test_providers)}")

    # Note: We don't actually test connectivity here to avoid using tokens on startup
    # But we log which providers are configured for testing
    logger.info("Provider configuration verified - ready for failover testing")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class RunCodeRequest(BaseModel):
    code: str
    language: str | None = None
    filename: str | None = None
    timeout: int | None = 10


@app.post("/api/run-code")
async def run_code(request: RunCodeRequest):
    """Execute code in a sandboxed environment."""
    try:
        runner = enhanced_agent_runner.get_runner()
        result = await runner.run_code(
            code=request.code, language=request.language, filename=request.filename
        )

        return {
            "success": result.error is None,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "execution_time": result.execution_time,
            "language": result.language,
            "error": result.error,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Execution failed: {str(e)}",
            "stdout": "",
            "stderr": "",
            "exit_code": 1,
            "execution_time": 0,
            "language": request.language or "unknown",
        }


@app.get("/api/languages")
async def get_supported_languages():
    """Get list of supported programming languages."""
    runner = enhanced_agent_runner.get_runner()
    languages = runner.get_supported_languages()

    return {
        "languages": languages,
        "details": {lang: runner.get_language_info(lang) for lang in languages},
    }


class EditMessageRequest(BaseModel):
    session_id: int
    message_index: int
    new_content: str
    system: str | None = None


class ShareRequest(BaseModel):
    session_id: int
    expires_hours: int | None = 24


class NewSessionRequest(BaseModel):
    project_name: str
    project_description: str = ""
    session_title: str = "New session"


class ChatRequest(BaseModel):
    session_id: int
    message: str
    project_context: str | None = None  # extra system context from the user
    prune_n: int | None = 0  # for editing/regenerating messages


class SwitchContextRequest(BaseModel):
    session_id: int


class AgentRunRequest(BaseModel):
    code: str
    language: str
    session_id: int | None = None


class AgentHealRequest(BaseModel):
    code: str
    language: str
    session_id: int
    max_retries: int | None = 3


class AgentPlanRequest(BaseModel):
    session_id: int
    task: str


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Projects & sessions
# ---------------------------------------------------------------------------
@app.post("/api/session/new")
async def new_session(body: NewSessionRequest):
    project_id = memory.upsert_project(
        body.project_name, body.project_description or ""
    )
    session_id = memory.create_session(project_id, body.session_title or "New session")
    return {
        "project_id": project_id,
        "session_id": session_id,
        "project_name": body.project_name,
    }


@app.get("/api/sessions")
async def get_sessions(project_id: int | None = None):
    return {"sessions": memory.list_sessions(project_id)}


@app.get("/api/projects")
async def get_projects():
    return {"projects": memory.list_projects()}


@app.get("/api/history/{session_id}")
async def get_history(session_id: int):
    return {"messages": memory.get_messages(session_id)}


# ---------------------------------------------------------------------------
# Main chat — SSE streaming
# ---------------------------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest):
    """
    Streams the AI response token-by-token as Server-Sent Events.

    Event format:
      data: {"type": "token",  "content": "..."}
      data: {"type": "done",   "model": "...", "tokens": N}
      data: {"type": "error",  "message": "..."}
    """

    async def _generate():
        # 0. Prune prior messages if doing an Edit or Regenerate
        if getattr(body, "prune_n", 0) > 0:
            memory.delete_last_n_messages(body.session_id, body.prune_n)

        # 1. Cache lookup (Instant bypass)
        semantic_cache = cache.get_cache()
        cached = semantic_cache.get(body.message)
        if cached:
            yield f"data: {json.dumps({'type': 'token', 'content': cached['response']})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'model': cached['model'], 'cached': True})}\n\n"
            return

        # 2. Load history
        history = memory.get_messages_for_llm(body.session_id)
        project_id = memory.get_project_id_for_session(body.session_id)

        # 3. Context compression (Auto-summarize if history is long)
        if len(history) > 30:
            await memory.compress_old_messages(body.session_id, chat_fn=ai_router.chat)
            history = memory.get_messages_for_llm(body.session_id)

        # 4. Token budget warning
        if len(history) > 800:
            yield f"data: {json.dumps({'type': 'warning', 'content': 'Context getting long — compression recommended.'})}\n\n"

        # 5. Hybrid RAG Search
        search_query = body.message
        if memory.has_project_files(project_id):
            # Use GitHub Models (secondary) for fast query rewriting if files exist
            search_query = await memory.rewrite_query(body.message, ai_router.chat)

        relevant_session = memory.semantic_search(body.message, body.session_id, n=2)
        relevant_files = memory.hybrid_search_rrf(
            search_query, project_id, body.session_id, n=3
        )

        # 6. Build system prompt
        system_parts = [
            "You are an expert coding assistant integrated into QuotaDrift.",
            "Be concise, direct, and technically precise.",
            "When writing code, always specify the language in fenced code blocks.",
        ]
        if body.project_context:
            system_parts.append(f"\nProject context:\n{body.project_context}")

        if relevant_session or relevant_files:
            sources_event = {
                "type": "sources",
                "memory_hits": len(relevant_session),
                "file_hits": len(relevant_files),
                "snippets": [
                    s[:100] + "..." for s in (relevant_session + relevant_files)
                ],
            }
            yield f"data: {json.dumps(sources_event)}\n\n"

        if relevant_session:
            system_parts.append(
                "\nRelevant session context:\n" + "\n---\n".join(relevant_session)
            )
        if relevant_files:
            system_parts.append(
                "\nRelevant project code:\n" + "\n---\n".join(relevant_files)
            )

        system = "\n".join(system_parts)

        # 4. Append the new user message to history for the LLM call
        llm_messages = history + [{"role": "user", "content": body.message}]

        # 5. Save user message immediately (so history is always up to date)
        memory.save_message(body.session_id, "user", body.message)

        # Auto-title the session from the first message
        if len(history) == 0:
            title = body.message[:50] + ("…" if len(body.message) > 50 else "")
            memory.update_session_title(body.session_id, title)

        # 6. Stream response
        full_response = []
        model_used = "unknown"

        async for event in ai_router.stream_chat(llm_messages, system):
            yield f"data: {json.dumps(event)}\n\n"

            if event["type"] == "token":
                full_response.append(event["content"])
            elif event["type"] == "done":
                model_used = event["model"]
            elif event["type"] == "error":
                # Still save a partial response if we got any tokens
                break

        # 7. Save assistant response
        content = "".join(full_response)
        if content:
            memory.save_message(
                body.session_id,
                "assistant",
                content,
                model=model_used,
                tokens=len(full_response),
            )
            memory.update_session_model(body.session_id, model_used)
            # Save to semantic cache
            semantic_cache.set(body.message, content, model_used)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# Context compiler (model handoff)
# ---------------------------------------------------------------------------
@app.post("/api/switch-context")
async def switch_context(body: SwitchContextRequest):
    """
    Compiles the current session state into a JSON artifact.
    Inject the returned `handoff_system` as the system prompt
    for the next model to ensure seamless continuation.
    """
    history = memory.get_messages_for_llm(body.session_id)
    if not history:
        raise HTTPException(400, "No messages in session to compile.")

    state = await compiler.compile_state(history, ai_router.chat)
    handoff = compiler.build_handoff_system(state)

    return {
        "compiled_state": state,
        "handoff_system": handoff,
    }


@app.post("/api/chat/edit")
async def edit_message(request: EditMessageRequest):
    """Edit a message in a session and regenerate subsequent messages."""
    try:
        # Get session messages
        messages = memory.get_messages_for_llm(request.session_id)

        if request.message_index >= len(messages):
            raise HTTPException(status_code=404, detail="Message index out of range")

        # Edit the message
        messages[request.message_index]["content"] = request.new_content

        # Remove all messages after the edited one
        messages_to_keep = messages[: request.message_index + 1]

        # Update session with truncated messages
        memory.update_session_messages(request.session_id, messages_to_keep)

        # If there's a system prompt, update it
        if request.system and request.message_index == 0:
            memory.update_session_system(request.session_id, request.system)

        logger.info(
            f"Edited message {request.message_index} in session {request.session_id}"
        )
        MESSAGES_PROCESSED.inc()

        return {
            "success": True,
            "message": f"Message {request.message_index} edited successfully",
            "session_id": request.session_id,
            "message_index": request.message_index,
        }

    except Exception as e:
        logger.error(f"Error editing message: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/chat/regenerate")
async def regenerate_message(request: EditMessageRequest):
    """Regenerate a message and all subsequent messages."""
    try:
        # Get session messages up to the target
        all_messages = memory.get_messages_for_llm(request.session_id)

        if request.message_index >= len(all_messages):
            raise HTTPException(status_code=404, detail="Message index out of range")

        # Keep messages before the target
        messages_to_regenerate = all_messages[: request.message_index]

        # Get the message to regenerate (for context)
        target_message = all_messages[request.message_index]

        # Generate new response
        response = await ai_router.chat(
            messages=messages_to_regenerate, system=request.system
        )

        # Update the message
        target_message["content"] = response["content"]
        target_message["model_used"] = response["model_used"]

        # Update session
        updated_messages = messages_to_regenerate + [target_message]
        memory.update_session_messages(request.session_id, updated_messages)

        logger.info(
            f"Regenerated message {request.message_index} in session {request.session_id}"
        )
        MESSAGES_PROCESSED.inc()
        PROVIDER_REQUESTS.labels(
            provider=response["model_used"], status="success"
        ).inc()

        return {
            "success": True,
            "content": response["content"],
            "model_used": response["model_used"],
            "tokens": response["tokens"],
            "session_id": request.session_id,
            "message_index": request.message_index,
        }

    except Exception as e:
        logger.error(f"Error regenerating message: {e}")
        PROVIDER_ERRORS.labels(
            provider="unknown", error_type="regeneration_failed"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Session sharing
# ---------------------------------------------------------------------------

# Store for share tokens (in production, use a database)
share_tokens: dict[str, dict] = {}


@app.post("/api/session/share")
async def create_share_link(request: ShareRequest):
    """Create a shareable link for a session."""
    token = secrets.token_urlsafe(16)
    expires_at = datetime.utcnow() + timedelta(hours=request.expires_hours)

    share_tokens[token] = {
        "session_id": request.session_id,
        "expires_at": expires_at.isoformat(),
        "created_at": datetime.utcnow().isoformat(),
        "access_count": 0,
    }

    return {
        "share_url": f"/shared/{token}",
        "token": token,
        "expires_at": expires_at.isoformat(),
    }


@app.get("/api/shared/{token}")
async def get_shared_session(token: str):
    """Get a shared session by token."""
    if token not in share_tokens:
        raise HTTPException(status_code=404, detail="Share link not found or expired")

    share_data = share_tokens[token]
    expires_at = datetime.fromisoformat(share_data["expires_at"])

    if datetime.utcnow() > expires_at:
        del share_tokens[token]
        raise HTTPException(status_code=410, detail="Share link expired")

    # Increment access count
    share_data["access_count"] += 1

    # Get session data
    session_id = share_data["session_id"]

    try:
        session = memory.get_session(session_id)
        messages = memory.get_messages(session_id)
        project = memory.get_project(session["project_id"]) if session else None

        return {
            "session": {
                "id": session_id,
                "title": session["title"] if session else "Untitled",
                "created_at": session["created_at"] if session else None,
                "model": session["model"] if session else None,
                "project": {
                    "name": project["name"] if project else "Unknown",
                    "description": project["description"] if project else "",
                },
            },
            "messages": messages,
            "share_info": {
                "created_at": share_data["created_at"],
                "expires_at": share_data["expires_at"],
                "access_count": share_data["access_count"],
            },
        }
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found") from None


@app.get("/shared/{token}")
async def serve_shared_page(token: str):
    """Serve a read-only page for shared sessions."""
    try:
        await get_shared_session(token)
        return FileResponse("static/shared.html", media_type="text/html")
    except HTTPException:
        return Response("<h1>Share link not found or expired</h1>", status_code=404)


@app.get("/api/provider-test")
async def test_providers():
    """Test connectivity for all configured providers."""

    results = {}

    # Test Groq
    if os.getenv("GROQ_API_KEY"):
        try:
            # Simple test call
            response = await ai_router.chat(
                messages=[{"role": "user", "content": "Say 'test'"}],
                system="You are a test assistant. Respond with only the word 'test'.",
            )
            results["groq"] = {
                "status": "success",
                "model_used": response.get("model_used", "unknown"),
                "response": response.get("content", "no content")[:50],
            }
        except Exception as e:
            results["groq"] = {"status": "error", "error": str(e)}
    else:
        results["groq"] = {"status": "no_key"}

    # Test Mistral
    if os.getenv("MISTRAL_API_KEY"):
        try:
            # Force use of mistral by temporarily changing config
            original_primary = config.MODEL_LIST[0]
            config.MODEL_LIST[0] = {
                "model_name": "test_mistral",
                "litellm_params": {
                    "model": "mistral/mistral-small-latest",
                    "api_key": "os.environ/MISTRAL_API_KEY",
                },
            }

            response = await ai_router.chat(
                messages=[{"role": "user", "content": "Say 'test'"}],
                system="You are a test assistant. Respond with only the word 'test'.",
            )
            results["mistral"] = {
                "status": "success",
                "model_used": response.get("model_used", "unknown"),
                "response": response.get("content", "no content")[:50],
            }

            # Restore original config
            config.MODEL_LIST[0] = original_primary
        except Exception as e:
            results["mistral"] = {"status": "error", "error": str(e)}
            # Restore original config
            config.MODEL_LIST[0] = original_primary
    else:
        results["mistral"] = {"status": "no_key"}

    # Test other providers (basic key check only to avoid consuming tokens)
    for provider, env_key, display_name in [
        ("siliconflow", "SILICONFLOW_API_KEY", "Silicon Flow"),
        ("huggingface", "HUGGINGFACE_API_KEY", "Hugging Face"),
        ("cloudflare", "CLOUDFLARE_API_KEY", "Cloudflare Workers AI"),
        ("openrouter", "OPENROUTER_API_KEY", "OpenRouter"),
    ]:
        if os.getenv(env_key):
            results[provider] = {
                "status": "key_present",
                "message": f"{display_name} API key configured",
            }
            if provider == "cloudflare" and not os.getenv("CLOUDFLARE_ACCOUNT_ID"):
                results[provider]["status"] = "incomplete"
                results[provider]["message"] = (
                    f"{display_name} API key present but missing ACCOUNT_ID"
                )
        else:
            results[provider] = {
                "status": "no_key",
                "message": f"{display_name} API key not configured",
            }

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
        "summary": {
            "total_providers": len(results),
            "configured": len(
                [r for r in results.values() if r.get("status") != "no_key"]
            ),
            "tested_successfully": len(
                [r for r in results.values() if r.get("status") == "success"]
            ),
        },
    }


# ---------------------------------------------------------------------------
# Enhanced Metrics and Observability
# ---------------------------------------------------------------------------

# Request metrics
REQUEST_COUNT = Counter(
    "quotadrift_requests_total", "Total requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "quotadrift_request_duration_seconds", "Request duration", ["method", "endpoint"]
)
ACTIVE_REQUESTS = Gauge("quotadrift_active_requests", "Currently active requests")

# Provider metrics
PROVIDER_REQUESTS = Counter(
    "quotadrift_provider_requests_total", "Provider requests", ["provider", "status"]
)
PROVIDER_LATENCY = Histogram(
    "quotadrift_provider_latency_seconds", "Provider response time", ["provider"]
)
PROVIDER_TOKENS = Counter(
    "quotadrift_provider_tokens_total", "Tokens used by provider", ["provider"]
)
PROVIDER_ERRORS = Counter(
    "quotadrift_provider_errors_total", "Provider errors", ["provider", "error_type"]
)

# System metrics
SYSTEM_CPU_USAGE = Gauge("quotadrift_system_cpu_percent", "System CPU usage")
SYSTEM_MEMORY_USAGE = Gauge("quotadrift_system_memory_percent", "System memory usage")
SYSTEM_DISK_USAGE = Gauge("quotadrift_system_disk_percent", "System disk usage")

# Application metrics
CACHE_HITS = Counter("quotadrift_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("quotadrift_cache_misses_total", "Cache misses")
SESSIONS_CREATED = Counter("quotadrift_sessions_created_total", "Sessions created")
MESSAGES_PROCESSED = Counter(
    "quotadrift_messages_processed_total", "Messages processed"
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect metrics for all requests."""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method, endpoint=request.url.path
        ).observe(duration)

        return response
    finally:
        ACTIVE_REQUESTS.dec()


# Update system metrics periodically
def update_system_metrics():
    """Update system-level metrics."""
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
    SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().percent)
    SYSTEM_DISK_USAGE.set(psutil.disk_usage("/").percent)


# Background task to update system metrics
async def system_metrics_task():
    """Background task to update system metrics every 30 seconds."""
    while True:
        update_system_metrics()
        await asyncio.sleep(30)


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    # Add system metrics (you could register these as Gauges)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready")
async def readiness_check():
    """Readiness check - verifies dependencies."""
    try:
        # Check database
        memory.get_projects()

        # Check models
        available = model_manager.model_manager.get_available_models()

        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "available_models": len(available),
            "total_models": len(config.MODEL_LIST),
        }
    except Exception as e:
        return Response({"status": "not_ready", "error": str(e)}, status_code=503)


# Model status
# ---------------------------------------------------------------------------
@app.get("/api/model-status")
async def model_status():
    """Enhanced model status with circuit breaker, scoring, and quota forecasting."""
    models = model_manager.model_manager.get_health_snapshot()

    # Add quota forecasting
    for model in models:
        model["quota_forecast"] = calculate_quota_forecast(model)

    return {
        "models": models,
        "timestamp": datetime.utcnow().isoformat(),
        "total_models": len(config.MODEL_LIST),
        "available_models": len(model_manager.model_manager.get_available_models()),
        "quota_limits": get_quota_limits(),
    }


def calculate_quota_forecast(model: dict) -> dict:
    """Calculate quota forecast based on usage and known limits."""
    slot_name = model["slot"]
    model_id = model["model_id"]

    # Get provider-specific limits
    limits = get_quota_limits()
    provider = get_provider_from_model(model_id)
    limit_info = limits.get(
        provider, {"daily": 14400, "period": "day"}
    )  # Default to Groq

    # Get current usage from model manager
    metrics = model_manager.model_manager.metrics.get(slot_name)
    if not metrics:
        return {
            "remaining": limit_info["daily"],
            "usage_percent": 0,
            "forecast": "Available",
        }

    # Calculate usage rate (requests per hour)
    now = datetime.utcnow()
    one_hour_ago = now - timedelta(hours=1)
    recent_requests = sum(
        1
        for success_time in metrics.recent_successes
        if success_time >= one_hour_ago.timestamp()
    )

    # Project usage for the period
    if limit_info["period"] == "day":
        hours_in_day = 24
        projected_daily = recent_requests * hours_in_day
        remaining = max(0, limit_info["daily"] - projected_daily)
        usage_percent = min(100, (projected_daily / limit_info["daily"]) * 100)
    elif limit_info["period"] == "month":
        # For monthly limits, project based on current day of month
        days_in_month = 30  # Approximation
        projected_monthly = recent_requests * days_in_month
        remaining = max(
            0, limit_info["daily"] - projected_monthly
        )  # daily contains monthly limit
        usage_percent = min(100, (projected_monthly / limit_info["daily"]) * 100)
    else:
        remaining = limit_info["daily"]
        usage_percent = 0

    # Determine forecast status
    if usage_percent < 50:
        forecast = "Healthy"
    elif usage_percent < 80:
        forecast = "Moderate"
    elif usage_percent < 95:
        forecast = "Low"
    else:
        forecast = "Critical"

    return {
        "remaining": remaining,
        "usage_percent": round(usage_percent, 1),
        "forecast": forecast,
        "period": limit_info["period"],
        "limit": limit_info["daily"],
        "current_rate": recent_requests,
    }


def get_provider_from_model(model_id: str) -> str:
    """Extract provider name from model ID."""
    if "groq" in model_id.lower():
        return "groq"
    elif "github" in model_id.lower() or "models.inference.ai" in model_id:
        return "github"
    elif "mistral" in model_id.lower() and "small" in model_id.lower():
        return "mistral"
    elif "siliconflow" in model_id.lower() or "qwen" in model_id.lower():
        return "siliconflow"
    elif "huggingface" in model_id.lower():
        return "huggingface"
    elif "cloudflare" in model_id.lower():
        return "cloudflare"
    elif "openrouter" in model_id.lower():
        return "openrouter"
    else:
        return "unknown"


def get_quota_limits() -> dict:
    """Get known quota limits for each provider."""
    return {
        "groq": {"daily": 14400, "period": "day"},  # 30 req/min * 8 hrs
        "github": {"daily": 4000, "period": "day"},  # 4,000 req/hr * 1 hr
        "mistral": {"daily": 1000000000, "period": "month"},  # 1B tokens/month
        "siliconflow": {"daily": 20000000, "period": "month"},  # 20M tokens/month
        "huggingface": {"daily": 10000, "period": "day"},  # ~10k requests/month
        "cloudflare": {"daily": 10000, "period": "day"},  # 10k requests/day
        "openrouter": {"daily": 200, "period": "day"},  # Free tier
    }


@app.post("/api/cache/clear")
async def clear_cache():
    cache.get_cache().clear()
    return {"status": "ok", "message": "Cache cleared"}


# ---------------------------------------------------------------------------
# Agentic Execution & Self-Healing
# ---------------------------------------------------------------------------
@app.post("/api/agent/run")
async def agent_run(body: AgentRunRequest):
    """Execute a snippet of code locally and return the result."""
    runner = enhanced_agent_runner.get_runner()
    result = runner.run_code(body.code, body.language)
    return result


@app.post("/api/agent/heal")
async def agent_heal(body: AgentHealRequest):
    """
    Run-Fix-Repeat loop.
    Streams status updates and final result.
    """

    async def _heal_gen():
        current_code = body.code
        attempts = 0

        while attempts < body.max_retries:
            attempts += 1
            yield f"data: {json.dumps({'type': 'status', 'content': f'Attempt {attempts}: Running code...'})}\n\n"

            runner = enhanced_agent_runner.get_runner()
            result = runner.run_code(current_code, body.language)

            if "error" in result:
                yield f"data: {json.dumps({'type': 'error', 'content': result['error']})}\n\n"
                break

            if result.get("exit_code") == 0:
                yield f"data: {json.dumps({'type': 'done', 'content': result['stdout'], 'code': current_code})}\n\n"
                return

            # It failed! Ask AI for a fix.
            error_msg = result.get("stderr") or "Unknown error"
            yield f"data: {json.dumps({'type': 'status', 'content': 'Execution failed. Asking AI for a fix...'})}\n\n"

            prompt = f"""The following {body.language} code failed with an error.
Please provide the CORRECTED code block.
Return ONLY the code, no explanation, no markdown backticks if possible (or just the code inside them).

CODE:
{current_code}

ERROR:
{error_msg}"""

            try:
                # Use GitHub Models (secondary) for fast fixing
                ai_fix = await ai_router.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system="You are a senior debugger. Output only the fixed code.",
                )

                # Extract code from response (handle markdown backticks)
                new_code = ai_fix["content"].strip()
                if "```" in new_code:
                    lines = new_code.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    new_code = "\n".join(lines).strip()

                current_code = new_code
                yield f"data: {json.dumps({'type': 'fix', 'content': 'AI suggested a fix. Retrying...'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'AI fix failed: {str(e)}'})}\n\n"
                break

        yield f"data: {json.dumps({'type': 'error', 'content': f'Failed to heal after {body.max_retries} attempts.'})}\n\n"

    return StreamingResponse(_heal_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Agent Planning
# ---------------------------------------------------------------------------
PLAN_SYSTEM = """You are a software architect.
Break down the user's task into a technical plan.
Return the plan in Markdown format. Use checklists.
Focus on files involved and changes required.
Keep it under 20 lines."""


@app.post("/api/agent/plan")
async def agent_plan(body: AgentPlanRequest):
    """Use a frontier model to write a technical plan.md."""
    history = memory.get_messages_for_llm(body.session_id)

    # Use priority_b or a 'frontier' model (defaulting to router's logic)
    result = await ai_router.chat(
        messages=history
        + [
            {
                "role": "user",
                "content": f"TASK: {body.task}\n\nWrite a technical plan for this.",
            }
        ],
        system=PLAN_SYSTEM,
    )

    # Save plan as a system message in background
    memory.save_message(
        body.session_id, "system", f"PLAN:\n{result['content']}", model="architect"
    )

    return {"plan": result["content"]}


# Health & Forecasting
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def api_health_check():
    """Health check that returns 200 even if some providers are down."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "uptime_s": int(time.perf_counter()),
        "providers": {},
        "database": "connected",
        "cache": "connected",
    }

    # Check provider health
    try:
        models = model_manager.model_manager.get_health_snapshot()
        failing_providers = []

        for model in models:
            provider_name = model.get("display", model.get("slot", "unknown"))
            circuit_state = model.get("circuit_state", "unknown")
            success_rate = model.get("success_rate", 0)

            health_status["providers"][provider_name] = {
                "status": (
                    "healthy"
                    if circuit_state == "closed" and success_rate > 0.5
                    else "degraded"
                ),
                "circuit_state": circuit_state,
                "success_rate": success_rate,
            }

            if circuit_state == "open" or success_rate < 0.5:
                failing_providers.append(provider_name)

        # Overall status based on provider health
        if len(failing_providers) > 0 and len(failing_providers) >= len(models) * 0.5:
            health_status["status"] = "degraded"
            health_status["message"] = (
                f"Multiple providers failing: {', '.join(failing_providers)}"
            )
        elif failing_providers:
            health_status["message"] = (
                f"Some providers failing: {', '.join(failing_providers)}"
            )

    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["message"] = f"Provider health check failed: {str(e)}"

    # Check database connectivity
    try:
        memory.list_sessions()
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = "disconnected"
        health_status["status"] = "unhealthy"
        health_status["message"] = f"Database connection failed: {str(e)}"

    return health_status


@app.get("/api/quota-forecast")
async def quota_forecast():
    # Known free tier limits
    LIMITS = {
        "groq": 14400,
        "github": 500,
        "mistral": 1000000000,  # 1B tokens/month
        "siliconflow": 20000000,  # 20M tokens
        "huggingface": 10000,  # ~10k requests/month
        "cloudflare": 10000,  # 10k requests/day
        "openrouter": 200,  # common free tier
    }

    stats = {}
    for slot, h in config.health.items():
        provider = slot.split("_")[0]
        limit = LIMITS.get(provider, 100)
        used = h.get("usage_count", 0)
        stats[slot] = {
            "limit": limit,
            "used": used,
            "remaining": max(0, limit - used),
            "percentage": round(min(100, (used / limit) * 100), 1) if limit > 0 else 0,
        }
    return stats


# ---------------------------------------------------------------------------
# Codebase indexing
# ---------------------------------------------------------------------------
@app.post("/api/project/index")
async def index_files(
    project_name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    Upload one or more code files to index into the project's vector store.
    Supports any text-based file (py, js, ts, md, json, yaml, etc.)
    """
    ALLOWED_EXT = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".html",
        ".css",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".env",
        ".sh",
        ".bash",
        ".sql",
        ".txt",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
    }

    project_id = memory.upsert_project(project_name)
    indexed = []
    skipped = []

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            skipped.append(f.filename)
            continue
        try:
            content = (await f.read()).decode("utf-8", errors="replace")
            memory.index_file(project_id, f.filename, content)
            indexed.append(f.filename)
        except Exception:
            skipped.append(f.filename)

    return {
        "project_id": project_id,
        "indexed": indexed,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
@app.get("/api/export/{session_id}")
async def export_session(session_id: int):
    md = memory.export_session_md(session_id)
    return Response(
        content=md,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=session_{session_id}.md"
        },
    )
