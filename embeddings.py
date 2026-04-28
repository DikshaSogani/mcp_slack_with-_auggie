"""
Slack Embedding Engine — Qdrant + FastEmbed (BAAI/bge-small-en-v1.5)

Handles dense semantic vectorization, Qdrant local vector DB management,
and semantic search over Slack messages.

Key improvements over TF-IDF + FAISS:
  • True semantic (dense) embeddings via ONNX — no GPU or PyTorch required
  • HNSW approximate nearest-neighbor index for fast search at scale
  • Incremental add without full rebuilds (unlike TF-IDF vocabulary constraints)
  • Auto-persisted local Qdrant storage; no separate save/load of matrix files
  • Built-in chunker for long messages to maximise retrieval precision
"""

import json
import os
import sys
import time
import uuid
import requests
from datetime import datetime, timezone
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models as qmodels
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR        = Path(__file__).parent / "data"
QDRANT_PATH     = DATA_DIR / "qdrant"
LAST_INDEXED_PATH = DATA_DIR / "last_indexed.json"

COLLECTION_NAME  = "slack_messages"
EMBED_MODEL      = "BAAI/bge-small-en-v1.5"   # 384-dim, ONNX, fast & accurate
EMBED_DIM        = 384

# ── Text chunker ─────────────────────────────────────────────────────────────

_MAX_CHUNK_CHARS = 800
_CHUNK_OVERLAP   = 80


def _chunk_text(text: str) -> list[str]:
    """Split long texts into overlapping character chunks for better recall.

    Short messages (≤ MAX_CHUNK_CHARS) are returned as-is.
    """
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + _MAX_CHUNK_CHARS, len(text))
        chunks.append(text[start:end])
        start += _MAX_CHUNK_CHARS - _CHUNK_OVERLAP
    return chunks


def _make_point_id(ts: str, channel_id: str, chunk_idx: int = 0) -> str:
    """Generate a stable, unique UUID for a (message, chunk) pair."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{channel_id}:{ts}:{chunk_idx}"))


class SlackEmbeddingEngine:
    """Manages the full lifecycle: ingest → embed → index (Qdrant) → search."""

    def __init__(self, slack_token: str | None = None):
        self.metadata: list[dict] = []
        self.slack_client: WebClient | None = None

        if slack_token:
            self.slack_client = WebClient(token=slack_token)
            self.token = slack_token

        self.ts_to_metadata: dict[str, dict] = {}

        # Ensure data directory exists and open Qdrant (local persistent mode)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        self._qdrant = QdrantClient(path=str(QDRANT_PATH))

        # Lazy-loaded ONNX embedder (downloads model on first use)
        self._embedder: TextEmbedding | None = None

    def _get_embedder(self) -> TextEmbedding:
        """Return a cached TextEmbedding instance (downloads ONNX model once)."""
        if self._embedder is None:
            self._embedder = TextEmbedding(EMBED_MODEL)
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts; returns a list of float vectors."""
        embedder = self._get_embedder()
        return [v.tolist() for v in embedder.embed(texts)]

    def _embed_query(self, query: str) -> list[float]:
        """Embed a single query string (uses the query prefix if the model supports it)."""
        embedder = self._get_embedder()
        return next(embedder.query_embed([query])).tolist()

    def _ensure_collection(self):
        """Create the Qdrant collection if it does not yet exist."""
        try:
            self._qdrant.get_collection(COLLECTION_NAME)
        except Exception:
            self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=EMBED_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    # ── Vector count / readiness ─────────────────────────────────────────

    @property
    def vector_count(self) -> int:
        """Number of vectors currently stored in the collection."""
        try:
            info = self._qdrant.get_collection(COLLECTION_NAME)
            return info.points_count or 0
        except Exception:
            return 0

    def is_ready(self) -> bool:
        """True if the collection exists and has at least one vector."""
        return self.vector_count > 0

    # ── Index Building ───────────────────────────────────────────────────

    def build_index(self, messages_with_meta: list[dict]) -> int:
        """Build the Qdrant collection from scratch (drops any existing data).

        Each dict must have at least: text, channel_id, channel_name, user, ts.
        Returns the number of chunks indexed.
        """
        filtered = [m for m in messages_with_meta if m.get("text", "").strip()]
        if not filtered:
            print("⚠️ No messages to index.", file=sys.stderr)
            return 0

        # Drop & recreate collection for a clean full rebuild
        try:
            self._qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        print(f"🔄 Embedding {len(filtered)} messages with {EMBED_MODEL}…", file=sys.stderr)
        total = self._upsert_messages(filtered)
        self._rebuild_metadata_cache()
        print(f"✅ Qdrant collection built: {total} chunks indexed.", file=sys.stderr)
        return total

    def add_to_index(self, messages_with_meta: list[dict]) -> int:
        """Incrementally add new messages without rebuilding from scratch.

        With dense embeddings the vocabulary is fixed, so we can truly add
        new vectors without touching existing ones — a major improvement over
        the former TF-IDF approach that required a full rebuild every time.
        Returns the number of new chunks added.
        """
        filtered = [m for m in messages_with_meta if m.get("text", "").strip()]
        if not filtered:
            return 0

        if not self.is_ready():
            return self.build_index(filtered)

        count = self._upsert_messages(filtered)
        self._rebuild_metadata_cache()
        return count

    def _upsert_messages(self, messages: list[dict]) -> int:
        """Embed and upsert a list of message dicts; returns chunk count."""
        self._ensure_collection()

        texts, metadatas, ids = [], [], []

        for m in messages:
            ts_float = float(m["ts"])
            dt = datetime.fromtimestamp(ts_float, tz=timezone.utc).isoformat()
            chunks = _chunk_text(m["text"])

            for idx, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    "channel_id":   m["channel_id"],
                    "channel_name": m["channel_name"],
                    "user":         m.get("user", "unknown"),
                    "ts":           m["ts"],
                    "thread_ts":    m.get("thread_ts"),
                    "datetime":     dt,
                    "text":         m["text"],   # always store the full text
                })
                ids.append(_make_point_id(m["ts"], m["channel_id"], idx))

        if not texts:
            return 0

        # Generate dense embeddings via fastembed ONNX (batched)
        vectors = self._embed(texts)

        points = [
            qmodels.PointStruct(id=pid, vector=vec, payload=meta)
            for pid, vec, meta in zip(ids, vectors, metadatas)
        ]

        # Upsert in batches of 256 to stay memory-friendly
        batch_size = 256
        for i in range(0, len(points), batch_size):
            self._qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + batch_size],
            )
        return len(points)

    def _rebuild_metadata_cache(self):
        """Scroll Qdrant and rebuild the in-memory ts → metadata lookup."""
        all_points, offset = [], None
        while True:
            points, next_offset = self._qdrant.scroll(
                collection_name=COLLECTION_NAME,
                with_payload=True,
                limit=1000,
                offset=offset,
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        self.metadata = [p.payload for p in all_points]
        # Keep only one entry per ts (the first chunk is sufficient for lookup)
        self.ts_to_metadata = {}
        for m in self.metadata:
            ts = m.get("ts")
            if ts and ts not in self.ts_to_metadata:
                self.ts_to_metadata[ts] = m

    # ── Index Persistence ────────────────────────────────────────────────

    def save_index(self):
        """Confirm persistence.

        Qdrant local mode auto-persists every write — there is nothing to
        flush manually.  This method exists for API compatibility with
        callers that previously called save_index() after build/add.
        """
        count = self.vector_count
        if count == 0:
            print("⚠️ Collection is empty — nothing to confirm.", file=sys.stderr)
            return
        print(f"💾 Qdrant collection persisted: {count} vectors → {QDRANT_PATH}", file=sys.stderr)

    def load_index(self) -> bool:
        """Load the existing Qdrant collection and rebuild the metadata cache.

        Returns True if the collection exists and has at least one vector.
        """
        if not self.is_ready():
            print("ℹ️ No existing Qdrant collection found.", file=sys.stderr)
            return False

        self._rebuild_metadata_cache()
        print(f"📂 Qdrant collection loaded: {self.vector_count} vectors, "
              f"{len(self.ts_to_metadata)} unique messages.", file=sys.stderr)
        return True

    # ── Last Indexed Tracking ────────────────────────────────────────────

    def save_last_indexed(self, channel_timestamps: dict[str, str]):
        """Save the last indexed timestamp per channel."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(LAST_INDEXED_PATH, "w", encoding="utf-8") as f:
            json.dump(channel_timestamps, f, indent=2)

    def load_last_indexed(self) -> dict[str, str]:
        """Load the last indexed timestamp per channel."""
        if not LAST_INDEXED_PATH.exists():
            return {}
        with open(LAST_INDEXED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Semantic Search ──────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the Qdrant collection for messages most similar to the query.

        For each matched message the method gathers full context by combining:
          1. Thread replies  — all Slack replies to the matched message
          2. Channel context — surrounding messages posted nearby in the same
             channel (solutions often live as normal messages, not in threads)

        Uses cosine similarity over BAAI/bge-small-en-v1.5 dense embeddings.
        Returns a list of dicts with: text, channel_name, thread_ts, score.
        """
        if not query or not query.strip() or not self.is_ready():
            return []

        fetch_k = min(top_k * 5, self.vector_count)
        query_vec = self._embed_query(query)

        scored_points = self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=fetch_k,
        ).points

        seen_msgs, output = set(), []
        for r in scored_points:
            ts = r.payload.get("ts", "")
            thread_ts = r.payload.get("thread_ts")
            thread_id = thread_ts if thread_ts else ts
            channel_id = r.payload.get("channel_id")

            if ts in seen_msgs:
                continue

            # 1. Fetch thread messages (parent + all replies)
            thread_messages = self._fetch_thread_from_index(thread_id, channel_id)

            # 2. Fetch surrounding channel messages (window of normal messages)
            channel_context = self._fetch_channel_context(channel_id, ts)

            # 3. Merge — deduplicate by ts, thread messages take priority
            merged = {m["ts"]: m for m in thread_messages}
            for m in channel_context:
                merged.setdefault(m["ts"], m)
            all_messages = sorted(merged.values(), key=lambda m: float(m.get("ts", 0)))

            # 4. Mark every included message as seen so it won't repeat
            for m in all_messages:
                seen_msgs.add(m["ts"])

            combined_text = (
                self._build_context_summary(all_messages, focus_ts=ts)
                if len(all_messages) > 1
                else r.payload.get("text", "")
            )

            output.append({
                "text":         combined_text,
                "channel_name": r.payload.get("channel_name"),
                "channel_id":   channel_id,
                "thread_ts":    thread_id,
                "score":        r.score,
            })

            if len(output) >= top_k:
                break

        return output

    def _fetch_thread_from_index(self, thread_ts: str, channel_id: str) -> list[dict]:
        """Fetch all messages in a thread (parent + replies) from Qdrant.

        Matches points where ``ts == thread_ts`` (the parent) OR
        ``thread_ts == thread_ts`` (all replies).
        """
        try:
            points, _ = self._qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=qmodels.Filter(
                    should=[
                        qmodels.FieldCondition(
                            key="ts",
                            match=qmodels.MatchValue(value=thread_ts),
                        ),
                        qmodels.FieldCondition(
                            key="thread_ts",
                            match=qmodels.MatchValue(value=thread_ts),
                        ),
                    ]
                ),
                with_payload=True,
                limit=100,
            )
        except Exception:
            return []

        seen, messages = set(), []
        for p in points:
            ts = p.payload.get("ts")
            if ts and ts not in seen:
                seen.add(ts)
                messages.append(p.payload)

        return sorted(messages, key=lambda m: float(m.get("ts", 0)))

    def _fetch_channel_context(self, channel_id: str, ts: str, window: int = 2) -> list[dict]:
        """Return up to `window` messages before and after `ts` in the same channel.

        Uses the in-memory ts_to_metadata cache — no extra Qdrant calls needed.
        This surfaces solutions posted as normal channel messages (not in threads).
        """
        channel_msgs = sorted(
            [m for m in self.ts_to_metadata.values() if m.get("channel_id") == channel_id],
            key=lambda m: float(m.get("ts", 0)),
        )
        idx = next((i for i, m in enumerate(channel_msgs) if m.get("ts") == ts), None)
        if idx is None:
            return []
        start = max(0, idx - window)
        end = min(len(channel_msgs), idx + window + 1)
        return channel_msgs[start:end]

    def _build_context_summary(self, messages: list[dict], focus_ts: str = "") -> str:
        """Build a clean, readable summary of a conversation context.

        Strips all ingestion-time prefixes ([Issue], [Thread], [Reply]) and
        labels each message by its role: matched message, thread reply, or
        surrounding channel message.
        """
        parts = []
        for m in messages:
            text = m.get("text", "").strip()

            # Strip enrichment prefixes, keep only the core content
            if text.startswith("[Issue] "):
                text = text[len("[Issue] "):]
                if "\n[Reply] " in text:
                    text = text[: text.index("\n[Reply] ")]
            elif "[Reply] " in text:
                text = text.rsplit("[Reply] ", 1)[-1]
            elif text.startswith("[Thread] ") and "\n" in text:
                text = text.split("\n", 1)[-1]

            is_reply = m.get("thread_ts") and m.get("thread_ts") != m.get("ts")
            is_focus = m.get("ts") == focus_ts

            if is_focus:
                prefix = "🎯"
            elif is_reply:
                prefix = "💬"
            else:
                prefix = "►"

            parts.append(f"{prefix} {text}")

        return "\n".join(parts)

    # ── Slack Data Fetching ──────────────────────────────────────────────

    def fetch_all_channels(self) -> list[dict]:
        """Fetch all channels the bot has access to."""
        if not self.slack_client:
            raise ValueError("Slack client not initialized. Provide a slack_token.")

        channels = []
        cursor = None

        while True:
            try:
                response = self.slack_client.conversations_list(
                    limit=200,
                    cursor=cursor,
                    types="public_channel"
                )
                channels.extend(response["channels"])
                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
            except SlackApiError as e:
                if e.response["error"] == "ratelimited":
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    print(f"⏳ Rate limited. Waiting {retry_after}s...", file=sys.stderr)
                    time.sleep(retry_after)
                else:
                    raise

        return channels

    def fetch_channel_messages(
        self,
        channel_id: str,
        oldest: str | None = None
    ) -> list[dict]:
        """Fetch all messages from a channel, handling pagination. Optionally only fetch messages newer than `oldest` timestamp."""
        if not self.slack_client:
            raise ValueError("Slack client not initialized. Provide a slack_token.")

        messages = []
        cursor = None

        while True:
            try:
                kwargs = {
                    "channel": channel_id,
                    "limit": 200,
                    "cursor": cursor,
                }
                if oldest:
                    kwargs["oldest"] = oldest

                response = self.slack_client.conversations_history(**kwargs)

                for msg in response.get("messages", []):
                    # Skip certain system messages but KEEP file sharing
                    subtype = msg.get("subtype")
                    if subtype and subtype not in ["file_share"]:
                        continue
                    messages.append(msg)

                    # Fetch thread replies if this message has a thread
                    if msg.get("reply_count", 0) > 0:
                        thread_msgs = self._fetch_thread_replies(
                            channel_id, msg["ts"], oldest
                        )
                        messages.extend(thread_msgs)

                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            except SlackApiError as e:
                if e.response["error"] == "ratelimited":
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    print(f"⏳ Rate limited. Waiting {retry_after}s...", file=sys.stderr)
                    time.sleep(retry_after)
                elif e.response["error"] == "not_in_channel":
                    print(f"  ⚠️ Bot not in channel {channel_id}, skipping.", file=sys.stderr)
                    break
                elif e.response["error"] == "channel_not_found":
                    print(f"  ⚠️ Channel {channel_id} not found, skipping.", file=sys.stderr)
                    break
                else:
                    raise

        return messages

    def _fetch_thread_replies(
        self,
        channel_id: str,
        thread_ts: str,
        oldest: str | None = None
    ) -> list[dict]:
        """Fetch replies in a thread, excluding the parent message."""
        replies = []
        cursor = None

        while True:
            try:
                kwargs = {
                    "channel": channel_id,
                    "ts": thread_ts,
                    "limit": 200,
                    "cursor": cursor,
                }
                if oldest:
                    kwargs["oldest"] = oldest

                response = self.slack_client.conversations_replies(**kwargs)

                for msg in response.get("messages", []):
                    # Skip the parent message (it's already captured)
                    if msg["ts"] == thread_ts:
                        continue
                    subtype = msg.get("subtype")
                    if subtype and subtype not in ["file_share"]:
                        continue
                    msg["thread_ts"] = thread_ts
                    replies.append(msg)

                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            except SlackApiError as e:
                if e.response["error"] == "ratelimited":
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    print(f"⏳ Rate limited on thread. Waiting {retry_after}s...", file=sys.stderr)
                    time.sleep(retry_after)
                else:
                    break

        return replies

    def download_file(self, url: str) -> bytes:
        """Download a private Slack file using the bot token."""
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.content

