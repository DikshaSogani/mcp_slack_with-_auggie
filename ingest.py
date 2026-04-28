"""
Slack Ingestion Script — Fetch messages and build/update the vector index.

Usage:
    python ingest.py              # Full re-index from scratch
    python ingest.py --update     # Incremental update (new messages only)

Thread-awareness
----------------
Thread replies are enriched with their parent message text before embedding.
This ensures that semantic search finds solutions even when the query describes
the *problem* (which lives in the parent) and the *solution* lives in a reply.
"""

import os
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

from embeddings import SlackEmbeddingEngine
import vision

load_dotenv()


def process_images_in_message(engine: SlackEmbeddingEngine, msg: dict) -> str:
    """Detect images, download them, and get descriptions from Augment Vision."""
    descriptions = []

    if "files" in msg:
        for f in msg["files"]:
            mimetype = f.get("mimetype", "")
            if mimetype.startswith("image/"):
                url = f.get("url_private")
                if not url:
                    continue

                print(f"   🖼️  Processing image: {f.get('name')}...", file=sys.stderr)
                try:
                    content = engine.download_file(url)
                    desc = vision.get_image_description(content, mimetype=mimetype)
                    if desc:
                        descriptions.append(f"[Image Description: {desc}]")
                except Exception as e:
                    print(f"   ⚠️  Image download failed: {e}", file=sys.stderr)

    return "\n".join(descriptions)


def enrich_thread_replies(
    messages: list[dict],
    existing_ts_map: dict | None = None,
) -> list[dict]:
    """Enrich both thread replies AND parent messages with full thread context.

    Replies are prefixed with the parent message so their vectors capture
    both the problem and the solution.

    Parent messages are suffixed with all their replies so that searching
    for the problem topic also surfaces the complete set of solutions.

    Parameters
    ----------
    messages:
        List of message dicts (already image-enriched) for the current batch.
        Each dict has keys: text, channel_id, channel_name, user, ts, thread_ts.
    existing_ts_map:
        Optional ts → metadata dict from the already-loaded engine index.
        Used in incremental mode so replies to *old* parent messages can still
        be enriched even when the parent is not in the current batch.

    Returns
    -------
    A new list of message dicts with enriched ``text`` fields.
    """
    # Build ts → text lookup from the current batch
    ts_to_text: dict[str, str] = {m["ts"]: m.get("text", "") for m in messages}

    # Build parent_ts → [reply texts] so parents can be enriched with replies
    thread_reply_texts: dict[str, list[str]] = {}
    for m in messages:
        thread_ts = m.get("thread_ts")
        if thread_ts and thread_ts != m["ts"]:
            thread_reply_texts.setdefault(thread_ts, []).append(m.get("text", ""))

    enriched = []
    for msg in messages:
        text = msg.get("text", "")
        thread_ts = msg.get("thread_ts")
        ts = msg["ts"]

        if thread_ts and thread_ts != ts:
            # ── Reply: prefix with parent context ──────────────────────────
            parent_text = ts_to_text.get(thread_ts, "")

            # Incremental mode: parent may be in the already-indexed collection
            if not parent_text and existing_ts_map:
                parent_meta = existing_ts_map.get(thread_ts, {})
                parent_text = parent_meta.get("text", "")

            if parent_text and parent_text.strip():
                text = f"[Thread] {parent_text}\n[Reply] {text}"
        else:
            # ── Parent: suffix with all its replies ─────────────────────────
            replies = thread_reply_texts.get(ts, [])
            if replies:
                reply_block = "\n".join(f"[Reply] {r}" for r in replies if r.strip())
                text = f"[Issue] {text}\n{reply_block}"

        enriched.append({**msg, "text": text})

    return enriched

def run_full_ingest(engine: SlackEmbeddingEngine):
    """Fetch ALL messages from all channels and build a fresh index."""

    print("\n" + "=" * 60, file=sys.stderr)
    print("   FULL INGESTION — Fetching all Slack messages", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    # Step 1: Get all channels
    print("📋 Fetching channel list...", file=sys.stderr)
    channels = engine.fetch_all_channels()
    print(f"   Found {len(channels)} channels.\n", file=sys.stderr)

    # Step 2: Fetch messages from each channel
    all_messages = []
    channel_timestamps = {}

    for i, channel in enumerate(channels):
        ch_id = channel["id"]
        ch_name = channel.get("name", "unknown")
        print(f"[{i+1}/{len(channels)}] 📥 Fetching: #{ch_name} ({ch_id})", file=sys.stderr)

        messages = engine.fetch_channel_messages(ch_id)
        print(f"   → {len(messages)} messages fetched.", file=sys.stderr)

        # Track the latest timestamp for incremental updates
        if messages:
            latest_ts = max(m["ts"] for m in messages)
            channel_timestamps[ch_id] = latest_ts

        # Enrich messages with channel info and process images
        for msg in messages:
            text = msg.get("text", "")

            # Add image descriptions to the text
            img_desc = process_images_in_message(engine, msg)
            if img_desc:
                text = f"{text}\n\n{img_desc}".strip()

            all_messages.append({
                "text": text,
                "channel_id": ch_id,
                "channel_name": ch_name,
                "user": msg.get("user", "unknown"),
                "ts": msg["ts"],
                "thread_ts": msg.get("thread_ts"),
            })

    print(f"\n📊 Total messages collected: {len(all_messages)}", file=sys.stderr)

    # Step 3: Enrich thread replies with parent context for richer embeddings
    thread_replies = sum(1 for m in all_messages if m.get("thread_ts") and m["thread_ts"] != m["ts"])
    print(f"🔗 Enriching {thread_replies} thread replies with parent context...", file=sys.stderr)
    all_messages = enrich_thread_replies(all_messages)

    # Step 4: Build index
    count = engine.build_index(all_messages)

    # Step 5: Save
    engine.save_index()
    engine.save_last_indexed(channel_timestamps)

    print(f"\n✅ Full ingestion complete! {count} messages indexed.\n")
    return count


def run_incremental_update(engine: SlackEmbeddingEngine):
    """Fetch only NEW messages since the last ingestion and add to the existing index."""

    print("\n" + "=" * 60, file=sys.stderr)
    print("   INCREMENTAL UPDATE — Fetching new messages only", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    # Load existing index
    loaded = engine.load_index()
    if not loaded:
        print("⚠️ No existing index found. Running full ingestion instead.\n", file=sys.stderr)
        return run_full_ingest(engine)

    existing_count = engine.vector_count
    print(f"📂 Existing collection has {existing_count} vectors.\n", file=sys.stderr)

    # Load last indexed timestamps
    last_indexed = engine.load_last_indexed()

    # Get all channels
    print("📋 Fetching channel list...", file=sys.stderr)
    channels = engine.fetch_all_channels()
    print(f"   Found {len(channels)} channels.\n", file=sys.stderr)

    # Fetch only new messages
    new_messages = []
    channel_timestamps = dict(last_indexed)  # start with existing

    for i, channel in enumerate(channels):
        ch_id = channel["id"]
        ch_name = channel.get("name", "unknown")
        oldest = last_indexed.get(ch_id)

        label = f"(since {oldest})" if oldest else "(first time)"
        print(f"[{i+1}/{len(channels)}] 📥 #{ch_name} {label}", file=sys.stderr)

        messages = engine.fetch_channel_messages(ch_id, oldest=oldest)

        if messages:
            # Filter out messages we already have (oldest is inclusive)
            if oldest:
                messages = [m for m in messages if m["ts"] > oldest]

            print(f"   → {len(messages)} new messages.", file=sys.stderr)

            latest_ts = max(m["ts"] for m in messages) if messages else oldest
            if latest_ts:
                channel_timestamps[ch_id] = latest_ts

            for msg in messages:
                text = msg.get("text", "")

                img_desc = process_images_in_message(engine, msg)
                if img_desc:
                    text = f"{text}\n\n{img_desc}".strip()

                new_messages.append({
                    "text": text,
                    "channel_id": ch_id,
                    "channel_name": ch_name,
                    "user": msg.get("user", "unknown"),
                    "ts": msg["ts"],
                    "thread_ts": msg.get("thread_ts"),
                })
        else:
            print(f"   → No new messages.", file=sys.stderr)

    if new_messages:
        print(f"\n📊 New messages to add: {len(new_messages)}", file=sys.stderr)
        # Enrich replies — parent may be in the current batch OR the existing index
        thread_replies = sum(1 for m in new_messages if m.get("thread_ts") and m["thread_ts"] != m["ts"])
        if thread_replies:
            print(f"🔗 Enriching {thread_replies} thread replies with parent context...", file=sys.stderr)
        new_messages = enrich_thread_replies(new_messages, engine.ts_to_metadata)
        added = engine.add_to_index(new_messages)
        engine.save_index()
        engine.save_last_indexed(channel_timestamps)
        print(f"\n✅ Incremental update complete! Added {added} messages.")
        print(f"   Total index size: {engine.vector_count} vectors.\n")
        return added
    else:
        print("\nℹ️ No new messages found. Index is up to date.\n")
        engine.save_last_indexed(channel_timestamps)
        return 0


def main():
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        print("❌ SLACK_BOT_TOKEN not found in .env file.", file=sys.stderr)
        sys.exit(1)

    engine = SlackEmbeddingEngine(slack_token=token)

    if "--update" in sys.argv:
        run_incremental_update(engine)
    else:
        run_full_ingest(engine)


if __name__ == "__main__":
    main()