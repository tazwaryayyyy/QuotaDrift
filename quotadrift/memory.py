"""
Persistent memory layer for the Multi-AI Switchboard.

SQLite  → structured storage (projects, sessions, messages)
ChromaDB → semantic vector store (RAG over conversation history)
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "switchboard.db"
CHROMA_DIR = BASE_DIR / "chroma_store"

# ---------------------------------------------------------------------------
# Lazy singletons — only loaded once per process
# ---------------------------------------------------------------------------
_state: dict[str, object | None] = {
    "embedder": None,
    "chroma_col": None,
}
# project_id -> HybridSearcher
_hybrid_searchers: dict[int, "HybridSearcher"] = {}


def _get_embedder() -> SentenceTransformer:
    embedder = _state["embedder"]
    if embedder is None:
        # 80MB model, CPU-friendly, good for semantic similarity
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _state["embedder"] = embedder
    return embedder  # type: ignore[return-value]


def _get_collection():
    collection = _state["chroma_col"]
    if collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(
            name="switchboard_memory",
            metadata={"hnsw:space": "cosine"},
        )
        _state["chroma_col"] = collection
    return collection


class HybridSearcher:
    def __init__(self, project_id: int):
        self.project_id = project_id
        self._corpus: list[str] = []
        self._meta: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._load_from_db()

    def _load_from_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT filename, content FROM project_files WHERE project_id=?",
                (self.project_id,),
            ).fetchall()
            for filename, content in rows:
                self.add_local(content, {"filename": filename})

    def add_local(self, text: str, meta: dict):
        self._corpus.append(text)
        self._meta.append(meta)
        tokenized_corpus = [re.findall(r"\w+", d.lower())
                            for d in self._corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, n: int = 5) -> list[dict]:
        if not self._bm25:
            return []
        tokens = re.findall(r"\w+", query.lower())
        scores = self._bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)),
                         key=lambda i: scores[i], reverse=True)[:n]
        return [
            {"text": self._corpus[i],
                "meta": self._meta[i], "score": scores[i]}
            for i in top_idx
        ]


def _get_hybrid_searcher(project_id: int) -> HybridSearcher:
    if project_id not in _hybrid_searchers:
        _hybrid_searchers[project_id] = HybridSearcher(project_id)
    return _hybrid_searchers[project_id]


# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS projects (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                description TEXT    DEFAULT '',
                created_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL REFERENCES projects(id),
                title       TEXT    DEFAULT 'New session',
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL,
                last_model  TEXT    DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                model       TEXT    DEFAULT '',
                tokens      INTEGER DEFAULT 0,
                timestamp   TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS project_files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL REFERENCES projects(id),
                filename    TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                indexed_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS provider_outcomes (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id        TEXT    NOT NULL,
                session_id        INTEGER NOT NULL REFERENCES sessions(id),
                strategy          TEXT    NOT NULL,
                selected_providers TEXT   NOT NULL,
                winner_provider   TEXT    DEFAULT '',
                success           INTEGER NOT NULL,
                latency_ms        INTEGER NOT NULL,
                tokens            INTEGER DEFAULT 0,
                cost_usd          REAL    DEFAULT 0,
                contract_met      INTEGER NOT NULL,
                fallback_triggered INTEGER NOT NULL,
                error             TEXT    DEFAULT '',
                created_at        TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
            CREATE INDEX IF NOT EXISTS idx_outcomes_provider_time ON provider_outcomes(winner_provider, created_at DESC);
        """)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------
def upsert_project(name: str, description: str = "") -> int:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO projects (name, description, created_at) VALUES (?,?,?)",
            (name, description, _now()),
        )
        row = conn.execute(
            "SELECT id FROM projects WHERE name=?", (name,)).fetchone()
        return row[0]


def list_projects() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, name, description, created_at FROM projects ORDER BY id DESC"
        ).fetchall()
    return [
        {"id": r[0], "name": r[1], "description": r[2], "created_at": r[3]}
        for r in rows
    ]


def get_projects() -> list[dict]:
    """Backward-compatible alias used by readiness checks."""
    return list_projects()


def get_project(project_id: int) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, name, description, created_at FROM projects WHERE id=?",
            (project_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "created_at": row[3],
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def create_session(project_id: int, title: str = "New session") -> int:
    now = _now()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO sessions (project_id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (project_id, title, now, now),
        )
        return cur.lastrowid


def update_session_title(session_id: int, title: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
            (title[:60], _now(), session_id),
        )


def update_session_model(session_id: int, model: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET last_model=?, updated_at=? WHERE id=?",
            (model, _now(), session_id),
        )


def update_session_system(session_id: int, system_prompt: str):
    """Update the system prompt for a session."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET system_prompt=?, updated_at=? WHERE id=?",
            (system_prompt, _now(), session_id),
        )


def update_session_messages(session_id: int, messages: list[dict]):
    """Update session messages, replacing all existing messages."""
    with sqlite3.connect(DB_PATH) as conn:
        # Delete existing messages
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))

        # Insert new messages
        for msg in messages:
            save_message(
                session_id=session_id,
                role=msg["role"],
                content=msg["content"],
                model=msg.get("model", ""),
                tokens=msg.get("tokens", 0),
            )


def list_sessions(project_id: int | None = None) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        if project_id:
            rows = conn.execute(
                """SELECT s.id, s.title, s.created_at, s.updated_at, s.last_model,
                          p.name as project_name
                   FROM sessions s JOIN projects p ON s.project_id=p.id
                   WHERE s.project_id=? ORDER BY s.updated_at DESC""",
                (project_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT s.id, s.title, s.created_at, s.updated_at, s.last_model,
                          p.name as project_name
                   FROM sessions s JOIN projects p ON s.project_id=p.id
                   ORDER BY s.updated_at DESC LIMIT 50"""
            ).fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "updated_at": r[3],
            "last_model": r[4],
            "project_name": r[5],
        }
        for r in rows
    ]


def get_session(session_id: int) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT id, project_id, title, created_at, updated_at, last_model
            FROM sessions
            WHERE id=?
            """,
            (session_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "project_id": row[1],
        "title": row[2],
        "created_at": row[3],
        "updated_at": row[4],
        "last_model": row[5],
        "model": row[5],
    }


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
def save_message(
    session_id: int,
    role: str,
    content: str,
    model: str = "",
    tokens: int = 0,
) -> int:
    ts = _now()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, model, tokens, timestamp) VALUES (?,?,?,?,?,?)",
            (session_id, role, content, model, tokens, ts),
        )
        msg_id = cur.lastrowid

    # Embed into ChromaDB for semantic retrieval (skip system messages)
    if role in ("user", "assistant") and content.strip():
        try:
            vec = _get_embedder().encode(content).tolist()
            _get_collection().add(
                documents=[content],
                embeddings=[vec],
                ids=[f"msg_{msg_id}"],
                metadatas=[
                    {
                        "session_id": session_id,
                        "role": role,
                        "model": model,
                        "timestamp": ts,
                    }
                ],
            )
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # Never crash on embedding failure

    return msg_id


def get_messages(session_id: int) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content, model, tokens, timestamp FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [
        {
            "role": r[0],
            "content": r[1],
            "model": r[2],
            "tokens": r[3],
            "timestamp": r[4],
        }
        for r in rows
    ]


def get_messages_for_llm(session_id: int) -> list[dict]:
    """Returns only role+content dicts suitable for LLM API calls."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in get_messages(session_id)
        if m["role"] in ("user", "assistant")
    ]


def semantic_search(query: str, session_id: int, n: int = 3) -> list[str]:
    """Find semantically similar past messages within this session."""
    try:
        vec = _get_embedder().encode(query).tolist()
        results = _get_collection().query(
            query_embeddings=[vec],
            n_results=n,
            where={"session_id": session_id},
        )
        return results["documents"][0] if results["documents"] else []
    except Exception:  # pylint: disable=broad-exception-caught
        return []


def hybrid_search_rrf(
    query: str, project_id: int, _session_id: int, n: int = 4
) -> list[str]:
    """
    Reciprocal Rank Fusion of vector search + BM25 for file RAG.
    """
    # 1. Vector results from files
    vec_results = search_project_files(query, project_id, n=n * 2)

    # 2. BM25 results from files
    bm25_results = _get_hybrid_searcher(project_id).search(query, n=n * 2)
    bm25_texts = [r["text"] for r in bm25_results]

    # 3. RRF Scoring
    scores: dict[str, float] = {}
    for rank, doc in enumerate(vec_results):
        scores[doc] = scores.get(doc, 0) + 1 / (60 + rank)
    for rank, doc in enumerate(bm25_texts):
        scores[doc] = scores.get(doc, 0) + 1 / (60 + rank)

    return sorted(scores, key=scores.get, reverse=True)[:n]


REWRITE_SYSTEM = """You are a search query optimizer for a code RAG system.
Rewrite the user's question into a precise search query that will retrieve the most relevant code snippets.
Return ONLY the rewritten query, nothing else. Max 15 words."""


async def rewrite_query(user_message: str, chat_fn) -> str:
    try:
        result = await chat_fn(
            messages=[{"role": "user", "content": user_message}],
            system=REWRITE_SYSTEM,
        )
        return result["content"].strip()
    except Exception:  # pylint: disable=broad-exception-caught
        return user_message


SUMMARIZE_SYSTEM = """Summarize this conversation segment into 3-5 bullet points.
Focus on: decisions made, code written, errors encountered, current status.
Be extremely concise. Output ONLY the bullets, no preamble."""


async def compress_old_messages(session_id: int, keep_recent: int = 10, chat_fn=None):
    """Summarize messages older than the last N, replace them with a summary."""
    if not chat_fn:
        return

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, role, content FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()

    msgs = [{"id": r[0], "role": r[1], "content": r[2]} for r in rows]
    if len(msgs) <= keep_recent + 5:
        return

    old = msgs[:-keep_recent]

    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}" for m in old)
    result = await chat_fn(
        messages=[{"role": "user", "content": convo_text}],
        system=SUMMARIZE_SYSTEM,
    )
    summary = result["content"]

    with sqlite3.connect(DB_PATH) as conn:
        old_ids = [m["id"] for m in old]
        conn.execute(
            f"DELETE FROM messages WHERE id IN ({','.join('?' for _ in old_ids)})",
            old_ids,
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, model, timestamp) VALUES (?,?,?,?,?)",
            (
                session_id,
                "system",
                f"[COMPRESSED HISTORY]\n{summary}",
                "summarizer",
                _now(),
            ),
        )


def delete_last_n_messages(session_id: int, n: int):
    """Deletes the last n messages from a specific session to support Edit / Regenerate features."""
    if n <= 0:
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM messages WHERE id IN (SELECT id FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?)",
            (session_id, n),
        )


# ---------------------------------------------------------------------------
# Project files (codebase indexing)
# ---------------------------------------------------------------------------
def index_file(project_id: int, filename: str, content: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO project_files (project_id, filename, content, indexed_at)
               VALUES (?,?,?,?)
               ON CONFLICT DO NOTHING""",
            (project_id, filename, content, _now()),
        )

    # Update hybrid searcher
    _get_hybrid_searcher(project_id).add_local(content, {"filename": filename})

    # Embed each file as a chunk for RAG
    try:
        vec = _get_embedder().encode(
            f"FILE: {filename}\n{content[:1000]}").tolist()
        col = _get_collection()
        doc_id = f"file_{project_id}_{filename}"
        col.upsert(
            documents=[f"FILE: {filename}\n{content}"],
            embeddings=[vec],
            ids=[doc_id],
            metadatas=[
                {"project_id": project_id, "filename": filename, "type": "file"}
            ],
        )
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def search_project_files(query: str, project_id: int, n: int = 3) -> list[str]:
    try:
        vec = _get_embedder().encode(query).tolist()
        results = _get_collection().query(
            query_embeddings=[vec],
            n_results=n,
            where={"$and": [{"project_id": project_id}, {"type": "file"}]},
        )
        return results["documents"][0] if results["documents"] else []
    except Exception:  # pylint: disable=broad-exception-caught
        return []


def has_project_files(project_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM project_files WHERE project_id=? LIMIT 1", (
                project_id,)
        ).fetchone()
        return row is not None


def get_project_id_for_session(session_id: int) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT project_id FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        return row[0] if row else 0


def export_session_md(session_id: int) -> str:
    msgs = get_messages(session_id)
    if not msgs:
        return "# Empty session\n"
    lines = ["# Switchboard Session Export\n"]
    for m in msgs:
        role = "**You**" if m["role"] == "user" else f"**AI** `{m['model']}`"
        lines.append(f"### {role}\n{m['content']}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Contract outcomes
# ---------------------------------------------------------------------------
def save_provider_outcome(record: dict):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO provider_outcomes (
                request_id,
                session_id,
                strategy,
                selected_providers,
                winner_provider,
                success,
                latency_ms,
                tokens,
                cost_usd,
                contract_met,
                fallback_triggered,
                error,
                created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record["request_id"],
                record["session_id"],
                record["strategy"],
                ",".join(record.get("selected_providers", [])),
                record.get("winner_provider") or "",
                1 if record.get("success") else 0,
                int(record.get("latency_ms", 0)),
                int(record.get("tokens", 0)),
                float(record.get("cost_usd", 0.0)),
                1 if record.get("contract_met") else 0,
                1 if record.get("fallback_triggered") else 0,
                record.get("error") or "",
                record.get("created_at", _now()),
            ),
        )


def get_provider_window_stats(provider_slot: str, window_size: int = 50) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT success, latency_ms, cost_usd
            FROM provider_outcomes
            WHERE winner_provider=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (provider_slot, window_size),
        ).fetchall()

    if not rows:
        return {
            "window_size": 0,
            "success_rate": 1.0,
            "avg_latency_ms": 0.0,
            "avg_cost_usd": 0.0,
        }

    size = len(rows)
    success_rate = sum(r[0] for r in rows) / size
    avg_latency = sum(r[1] for r in rows) / size
    avg_cost = sum(r[2] for r in rows) / size

    return {
        "window_size": size,
        "success_rate": round(success_rate, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_cost_usd": round(avg_cost, 6),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.utcnow().isoformat()
