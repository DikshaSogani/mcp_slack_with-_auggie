import os
import io
import sys
import contextlib
import requests
from dotenv import load_dotenv
from slack_sdk import WebClient
from mcp.server.fastmcp import FastMCP
from embeddings import SlackEmbeddingEngine
import vision

# load environment variables
load_dotenv()

# ── Slack client ──────────────────────────────────────────────────────────────
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

# ── MCP server ────────────────────────────────────────────────────────────────
mcp = FastMCP("slack-assistant")

# embedding engine — load Qdrant collection on startup
with contextlib.redirect_stdout(sys.stderr):
    engine = SlackEmbeddingEngine()
    _index_loaded = engine.load_index()
    if _index_loaded:
        print(f"✅ Semantic search ready — {engine.vector_count} vectors loaded.")
    else:
        print("⚠️ No Qdrant collection found. Run 'python ingest.py' first to enable semantic search.")


def _describe_image(file_info: dict) -> str:
    """Return a description of an image using the Augment Vision API.
    Matches the indexing path in ingest.py so live reads produce the same
    quality of description as semantic_search results. Falls back to the
    filename on any failure, and logs the reason to stderr."""
    url   = file_info.get("url_private", "")
    name  = file_info.get("name", "image")
    mime  = file_info.get("mimetype", "")
    token = os.getenv("SLACK_BOT_TOKEN", "")
    if not url or not token:
        return name
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=15)
        resp.raise_for_status()
        desc = vision.get_image_description(resp.content, mimetype=mime)
        if desc and not desc.startswith("[Image description failed"):
            return desc.strip()
        print(f"⚠️  vision returned no description for {name}: {desc}", file=sys.stderr)
        return name
    except Exception as e:
        print(f"⚠️  _describe_image failed for {name}: {e}", file=sys.stderr)
        return name


# TOOL 1 — READ MESSAGES
@mcp.tool()
async def read_messages(channel_id: str, limit: int = 5):
    """Read the last `limit` messages from a channel.
    Thread replies are shown indented under their parent.
    Image attachments are described in one line using OCR."""

    response = client.conversations_history(channel=channel_id, limit=limit)

    messages = []
    for i, msg in enumerate(response.get("messages", [])):
        ts   = msg.get("ts")
        text = msg.get("text", "")

        # Describe any attached images
        for f in msg.get("files", []):
            if f.get("mimetype", "").startswith("image/"):
                desc = _describe_image(f)
                text = (text + f"\n  🖼️  Image: {desc}").strip()

        # Fall back to knowledge-base enriched text if available
        if not text and ts in engine.ts_to_metadata:
            text = engine.ts_to_metadata[ts].get("text", "")

        line = f"[{i+1}] {text}"

        # Fetch and append thread replies
        if msg.get("reply_count", 0) > 0:
            try:
                thread = client.conversations_replies(channel=channel_id, ts=ts)
                for reply in thread.get("messages", [])[1:]:   # skip parent
                    reply_text = reply.get("text", "").strip()
                    for rf in reply.get("files", []):
                        if rf.get("mimetype", "").startswith("image/"):
                            reply_text += f"\n      🖼️  Image: {_describe_image(rf)}"
                    if reply_text:
                        line += f"\n    ↳ {reply_text}"
            except Exception:
                pass

        messages.append(line)

    if not messages:
        return "No messages found in this channel."

    return "\n".join(messages)


# TOOL 2 — SEND MESSAGE
@mcp.tool()
async def send_message(channel_id: str, text: str):
    """Send a message to a specific Slack channel."""
    try:
        response = client.chat_postMessage(
            channel=channel_id,
            text=text
        )
        return {"status": "success", "message": response["message"]["text"]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# TOOL 3 — SEMANTIC SEARCH across all Slack channels
@mcp.tool()
async def semantic_search(query: str, top_k: int = 10, min_score: float = 0.55):
    """Search all indexed Slack messages for content semantically similar to the query.
    Use this tool when the user asks about a problem, wants to find past discussions,
    or needs solutions based on team knowledge shared in Slack channels.
    Results are grouped by thread so both the issue and all replies/solutions
    are returned together as one combined result.

    Each result is a block of lines prefixed with:
      🎯  the matched message (the question / topic)
      💬  a thread reply (the answer / solution)
      ►   surrounding channel context near the match

    Low-relevance hits below `min_score` (default 0.55) are dropped so the
    caller is not tempted to hallucinate from noisy matches."""

    if not engine.is_ready():
        return "No index available. Run 'python ingest.py' to build the index first."

    raw = engine.search(query, top_k=top_k)
    results = [r for r in raw if r.get("score", 0) >= min_score]

    if not results:
        best = max((r.get("score", 0) for r in raw), default=0)
        return (
            f"No relevant discussion found in Slack for: '{query}' "
            f"(best relevance was {best:.2f}, below min_score={min_score:.2f}). "
            f"Do not fabricate an answer — tell the user nothing matched."
        )

    lines = [f"🔍 Results for: '{query}'  (min_score={min_score:.2f})\n{'─' * 50}"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n[{i}] #{r.get('channel_name')}  (relevance: {r.get('score', 0):.3f})")
        lines.append(r.get("text", ""))
    dropped = len(raw) - len(results)
    if dropped:
        lines.append(f"\n({dropped} lower-relevance result(s) hidden.)")
    lines.append("─" * 50)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Slack MCP Server running...", file=sys.stderr)
    mcp.run()
