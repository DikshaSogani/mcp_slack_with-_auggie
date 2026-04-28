"""
CLI wrapper for semantic search over indexed Slack messages.

Usage:
    python search_cli.py "your query here"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from embeddings import SlackEmbeddingEngine


def _clean(text: str) -> str:
    """Strip ingestion prefixes that may have leaked into standalone results."""
    for prefix in ("[Issue] ", "[Thread] ", "[Reply] "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            if "\n[Reply] " in text:
                text = text[: text.index("\n[Reply] ")]
            break
    return text.strip()


def _extract_relevant_lines(text: str) -> str:
    """Keep only matched (🎯) and thread-reply (💬) lines — drop noisy context (►)."""
    if not any(p in text for p in ["🎯", "💬", "►"]):
        return _clean(text)

    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("🎯") or stripped.startswith("💬"):
            # Remove the emoji prefix for cleaner output
            content = stripped[2:].strip()
            if content and content not in kept:
                kept.append(content)
    return "\n".join(kept)


def main():
    if len(sys.argv) < 2:
        print("Usage: python search_cli.py \"your query\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    engine = SlackEmbeddingEngine()
    engine.load_index()

    if not engine.is_ready():
        print("❌ No index found. Run: python ingest.py")
        sys.exit(1)

    results = engine.search(query, top_k=5)

    # Keep only results above relevance threshold and deduplicate content
    MIN_SCORE = 0.70
    seen_texts, filtered = set(), []
    for r in results:
        if r.get("score", 0) < MIN_SCORE:
            continue
        relevant = _extract_relevant_lines(r.get("text", "").strip())
        if not relevant or relevant in seen_texts:
            continue
        seen_texts.add(relevant)
        filtered.append({**r, "text": relevant})

    if not filtered:
        print(f"\nNo relevant results found for: '{query}'")
        sys.exit(0)

    print(f"\n🔍 Results for: '{query}'")
    print("─" * 50)

    for i, r in enumerate(filtered, 1):
        channel = r.get("channel_name", "unknown")
        score = r.get("score", 0)
        print(f"\n[{i}] #{channel}  (relevance: {score:.2f})")
        print(r["text"])

    print("\n" + "─" * 50)


if __name__ == "__main__":
    main()
