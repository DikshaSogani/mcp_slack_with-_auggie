import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

load_dotenv()

# ── Auggie CLI integration ────────────────────────────────────────────────────
# This client shells out to the `auggie` CLI for every turn. Auggie itself acts
# as the MCP client, spawning server.py via the generated MCP config below, so
# no in-process MCP session is needed here.
#
# Prerequisites:
#   1. Install:       npm install -g @augmentcode/auggie
#   2. Authenticate:  auggie login   (or export AUGMENT_SESSION_AUTH)
#
# Optional .env keys:
#   AUGMENT_MODEL     — auggie short model id (default: sonnet4.5).
#                       See `auggie model list` for available ids.
#   MCP_PYTHON        — python used to launch server.py (default: current interpreter)
#   MCP_SERVER        — MCP server entrypoint (default: server.py)
#   AUGGIE_BIN        — override the auggie executable path
#   AUGGIE_MAX_TURNS  — cap agent turns per call (default: 15)

_AUGMENT_MODEL    = os.getenv("AUGMENT_MODEL", "sonnet4.5")
_AUGGIE_BIN       = os.getenv("AUGGIE_BIN", "auggie")
_AUGGIE_MAX_TURNS = os.getenv("AUGGIE_MAX_TURNS", "15")

if shutil.which(_AUGGIE_BIN) is None:
    print(f"❌ ERROR: '{_AUGGIE_BIN}' not found on PATH.")
    print("   Install it with:  npm install -g @augmentcode/auggie")
    print("   Then run:          auggie login")
    sys.exit(1)


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_mcp_config() -> str:
    """Write an ephemeral MCP config pointing at this project's server.py and
    return its path. Auggie loads it via --mcp-config."""
    cfg = {
        "mcpServers": {
            "slack": {
                "command": os.getenv("MCP_PYTHON", sys.executable),
                "args":    [os.getenv("MCP_SERVER", "server.py")],
            }
        }
    }
    fd, path = tempfile.mkstemp(prefix="mcp_slack_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return path


def run_auggie(prompt: str, mcp_config_path: str) -> str:
    """Run one non-interactive auggie turn and return its stdout."""
    cmd = [
        _AUGGIE_BIN,
        "--print",
        "--quiet",
        "--mcp-config", mcp_config_path,
        "--model", _AUGMENT_MODEL,
        "--max-turns", str(_AUGGIE_MAX_TURNS),
        prompt,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        return "❌ auggie CLI not found — install with: npm install -g @augmentcode/auggie"

    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or f"auggie exited with code {proc.returncode}"
        return f"❌ Auggie error: {err}"

    out = (proc.stdout or "").strip()
    # Strip auggie's leading model/runtime notices so the final reply is clean.
    cleaned = [
        line for line in out.splitlines()
        if not line.lstrip().startswith(("⚠️", "ℹ️"))
        and "Unknown model" not in line
        and "falling back to default model" not in line
    ]
    return "\n".join(cleaned).strip()


def _build_prompt(system: str, history: list, user_msg: str) -> str:
    """Compose a single prompt string. Auggie --print is stateless, so the
    recent transcript is prepended to keep the conversation coherent."""
    parts = [system, ""]
    if history:
        parts.append("Previous conversation:")
        for role, text in history[-8:]:  # last 4 exchanges
            parts.append(f"{role}: {text}")
        parts.append("")
    parts.append(f"User: {user_msg}")
    parts.append("Assistant:")
    return "\n".join(parts)


_SYSTEM_PROMPT = (
    "You are a Slack knowledge assistant with access to the 'slack' MCP server "
    "(tools: read_messages, send_message, semantic_search).\n"
    "AVAILABLE CHANNELS: {channels}.\n"
    "When the user mentions a channel by its Name, you MUST use its corresponding ID for tool calls.\n"
    "When the user asks about a problem or needs help finding information, "
    "use the semantic_search tool to find relevant past discussions from Slack channels. "
    "IMPORTANT: If you find multiple different solutions or answers discussed across different channels, "
    "you MUST mention all of them. Do not just summarize the first one you read. "
    "IMPORTANT: You must ONLY answer the user's specific question. Discard and ignore any search results "
    "or context that are unrelated to the specific topic requested. "
    "Always cite the channel name and the relevant raw text from the results. "
    "Use read_messages and send_message tools for direct channel operations. "
    "IMPORTANT: If the user asks you to read, list, or give messages from a channel, "
    "you MUST display the actual messages verbatim. Do not suppress or summarize them. "
    "Extract channel IDs directly from the user message or the AVAILABLE CHANNELS list when provided. "
    "\n"
    "SEARCH RESULT FORMAT: semantic_search returns blocks of lines prefixed with emoji markers:\n"
    "  🎯  = the matched message (typically the question / issue / topic the user is asking about)\n"
    "  💬  = a thread reply to the matched message (typically the answer / solution)\n"
    "  ►  = surrounding channel messages near the match (context — may or may not be relevant)\n"
    "When the user asks for a 'thread', 'solution', 'answer', 'reply', or 'fix' for a topic: "
    "quote the 🎯 line as the question and quote every 💬 line verbatim as the solution(s). "
    "If there are NO 💬 lines for a matching 🎯, say explicitly: 'No thread replies were found for this question.' "
    "Do NOT invent, paraphrase, generalize, or pull in external knowledge. Your answer must be "
    "grounded ONLY in the exact text returned by semantic_search / read_messages. "
    "If the tool returns no results above the relevance threshold, say 'No relevant discussion found in Slack for <topic>.' "
    "and stop — never fabricate a plausible-sounding answer.\n"
    "NOTE: Messages may also contain '[Image Description: ...]' tags generated by vision AI — treat these as faithful text content. "
    "Keep replies short and clear."
)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("   Slack AI Assistant  (Auggie CLI + MCP)")
    print("=" * 58)
    print("Examples:  read last 5 messages from C08XXXXXX")
    print("           send 'Hello!' to C08XXXXXX")
    print("           summarize messages from C08XXXXXX")
    print("Type 'exit' to quit.\n")

    # Load channel map (created by fetch_channels.py)
    channels_file = os.getenv("CHANNELS_JSON", "channels.json")
    try:
        with open(channels_file, "r", encoding="utf-8") as f:
            channels = json.load(f)
        channel_context_str = ", ".join(
            [f"Name: {k} -> ID: {v}" for k, v in channels.items()]
        )
    except FileNotFoundError:
        channel_context_str = "No channel mapping found."

    system_prompt = _SYSTEM_PROMPT.format(channels=channel_context_str)

    mcp_config_path = _write_mcp_config()
    print(f"✅ Ready — auggie will load MCP config from {mcp_config_path}\n")

    history: list = []
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("👋 Goodbye!")
                break

            prompt = _build_prompt(system_prompt, history, user_input)
            reply = run_auggie(prompt, mcp_config_path)

            print("\n💬 AI Final Reply:")
            print(f"🤖 {reply}\n")

            history.append(("User", user_input))
            history.append(("Assistant", reply))
    finally:
        try:
            os.unlink(mcp_config_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()




