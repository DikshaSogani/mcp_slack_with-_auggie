# 🧭 Slack MCP Agent — Workflow (in plain words)

This document walks through **how the project works** from the moment you type a question to the moment you see an answer. No jargon — just the path data takes and which file handles each step.

---

## 🎬 The one-minute summary

You talk to a terminal chatbot. The chatbot doesn't know anything about Slack on its own — it borrows three skills from a local helper process: *read Slack*, *send to Slack*, and *search past conversations by meaning*. An AI model (Claude, run through the `auggie` CLI) decides which skill to use for each of your questions, gathers the results, and writes the final answer.

There are **two pipelines** in this project:

1. **Offline pipeline (once / occasionally)** — download Slack history and turn it into a searchable knowledge base.
2. **Live pipeline (every time you ask something)** — AI reads your question, calls the right tool, and answers.

---

## 🗂️ Offline pipeline — building the knowledge base

Run: `python ingest.py`

```
Slack API ──► ingest.py ──► (for each message)
                              │
                              ├── text message? → keep as-is
                              └── image file?   → vision.py → auggie CLI → text description
                              │
                              ▼
                        enrich with thread context
                        (question + all replies together)
                              │
                              ▼
                      embeddings.py → FastEmbed (ONNX)
                              │
                              ▼
                        Qdrant vector DB (data/qdrant/)
```

**What happens and why:**

| Step | File | Simple explanation |
|---|---|---|
| 1. Fetch | `ingest.py` + `embeddings.py` | Downloads every message from every channel the bot is a member of, using Slack's Web API. |
| 2. Describe images | `vision.py` | Screenshots and diagrams are common in Slack. Instead of ignoring them, we send each image to `auggie` and ask "describe this in detail." The description becomes searchable text. |
| 3. Stitch threads | `ingest.py` | A question and its reply are usually in the same thread. We glue them together so searching for the question *also* surfaces the answer. |
| 4. Embed | `embeddings.py` (FastEmbed) | Each message is turned into a 384-number vector that captures its meaning. The model is `BAAI/bge-small-en-v1.5` — small, fast, CPU-only. |
| 5. Store | Qdrant (local) | Vectors are saved to `data/qdrant/` on disk. No external database server is needed — it's just files. |

`data/last_indexed.json` remembers the newest message timestamp per channel, so `python ingest.py --update` only fetches what's new.

---

## 💬 Live pipeline — answering your question

Run: `python aiclient.py` → you get a `You:` prompt.

```
You type a question
        │
        ▼
aiclient.py  ── wraps prompt with system instructions + short history
        │
        ▼
spawns:  auggie --print --mcp-config <temp-file>
        │
        ▼
🤖 auggie (Claude Sonnet 4.5) reads your question and decides:
        │
        ├── "Need recent messages?"     → call  read_messages
        ├── "Need to post something?"   → call  send_message
        └── "Need old/contextual info?" → call  semantic_search
        │
        ▼
server.py (FastMCP) runs the chosen tool:
        │
        ├── read_messages   → Slack API (conversations.history + replies)
        │                     + vision.py for any images found
        ├── send_message    → Slack API (chat.postMessage)
        └── semantic_search → embeddings.py → Qdrant similarity search
        │
        ▼
auggie gets the tool result, may call more tools, then writes a final answer
        │
        ▼
aiclient.py prints the answer under "Agent:"
```

**Key ideas in plain words:**

- **`aiclient.py` is just a chat loop.** It has no AI built in. It hands every prompt to `auggie` as a subprocess and prints whatever comes back.
- **`auggie` is the brain.** It is the Augment CLI that talks to Claude and knows the Model Context Protocol, so it can call external tools by itself.
- **`server.py` is the hands.** It exposes three tools over MCP. Auggie calls these tools the same way a person would call functions, but the "calls" happen over stdin/stdout pipes.
- **MCP is the phone line** between auggie and `server.py`. The temp config file tells auggie "here's how to start the Slack server and talk to it."

---

## 🧰 The three tools, explained simply

| Tool | What you'd ask for | What it actually does |
|---|---|---|
| `read_messages` | "Show me the last 10 messages in #demo" | Calls Slack's history endpoint, pulls thread replies too, and describes any images via `vision.py`. |
| `send_message` | "Send 'Hi team' to #demo" | Calls `chat.postMessage`. The agent always confirms before sending. |
| `semantic_search` | "What's the fix for the Victoria Metrics issue?" | Embeds your query, finds the closest vectors in Qdrant, and returns those messages with their thread context. Results below similarity **0.55** are dropped so the AI doesn't invent answers from weak matches. |

---

## 🤔 Why these particular tools?

- **Auggie CLI instead of OpenAI SDK** — the Augment tenant used here does not accept direct HTTP API calls (returns 404). The CLI is the only working path and it already handles auth, tool-calling, and streaming.
- **FastMCP** — a few lines of Python + a `@mcp.tool()` decorator is enough to expose a function; no web server, no REST, no schema files.
- **Qdrant + FastEmbed** — bundled into one `pip` package, runs fully local, no GPU, no PyTorch, no external DB. Good enough quality for Slack-sized corpora.
- **Auggie for vision too** — reusing the same CLI for image descriptions means one auth, one install, and identical quality between offline indexing and live reads.

---

## 🔁 End-to-end example

> **You:** *what was the conclusion on the grafana dashboard permissions thread?*

1. `aiclient.py` sends the prompt to `auggie`.
2. Auggie decides this is historical → calls `semantic_search("grafana dashboard permissions conclusion")`.
3. `server.py` → `embeddings.py` → Qdrant returns the top-matching thread (original question + all replies, score 0.78).
4. Auggie reads the thread text, sees the decision ("we granted Editor role to the on-call rotation"), and writes the answer.
5. `aiclient.py` prints:
   > **Agent:** The thread in #infra concluded that the on-call rotation would be granted the Editor role on the Grafana dashboards. @alice applied the change on 2026-03-14.

No hallucination, because the answer came from a real indexed message — and if the top match had scored below 0.55, the agent would have replied "I don't have that information" instead.

---

## 📌 TL;DR

- **`ingest.py`** fills the memory (offline).
- **`aiclient.py`** is the mouth you talk to.
- **`auggie`** is the brain that decides what to do.
- **`server.py`** is the hands that actually touch Slack and the vector DB.
- **Qdrant + FastEmbed** is the memory.
- **`vision.py`** turns pictures into searchable text.

Everything else (`fetch_channels.py`, `search_cli.py`) is a convenience script around these core pieces.






# Slack MCP Agent — Workflow

A terminal chatbot that reads, searches, and sends Slack messages using AI.
You type a question → an AI brain decides what to do → tools talk to Slack → you get an answer.

---

## The Big Picture

There are two separate pipelines in this project:

| Pipeline | When it runs | What it does |
|---|---|---|
| **Offline** | Once (or when you want to update) | Downloads Slack history and builds a searchable database |
| **Live** | Every time you ask a question | AI reads your question, picks the right tool, and answers |

---

## Files and Their Roles

| File | Role | Simple description |
|---|---|---|
| `fetch_channels.py` | Setup helper | Downloads channel names and saves them to `channels.json` |
| `ingest.py` | Offline pipeline | Fetches all Slack messages and builds the knowledge base |
| `embeddings.py` | Memory engine | Converts text into vectors and stores/searches them in Qdrant |
| `vision.py` | Image reader | Sends images to auggie and gets back a text description |
| `server.py` | MCP tool server | Exposes 3 tools over MCP that auggie can call |
| `aiclient.py` | Chat interface | The terminal prompt you talk to |
| `search_cli.py` | Dev utility | Lets you test semantic search from the command line |

---

## Offline Pipeline — Building the Knowledge Base

Run this once before you start chatting:

```
python fetch_channels.py    # save channel name → ID map
python ingest.py            # build the vector database
```

### What happens step by step

```
┌─────────────┐
│  Slack API  │  ← bot fetches all messages from all channels
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ingest.py  │  ← loops through every message
└──────┬──────┘
       │
       ├── Has an image? ──► vision.py ──► auggie CLI ──► text description
       │
       ├── Has thread replies? ──► glue parent + replies into one block
       │
       ▼
┌──────────────────┐
│  embeddings.py   │  ← converts text into 384-number vectors
│  (FastEmbed ONNX)│     model: BAAI/bge-small-en-v1.5
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│  Qdrant DB   │  ← saves vectors to disk at data/qdrant/
│  (local)     │     no server needed — just files
└──────────────┘
       │
       ▼
┌─────────────────────┐
│  last_indexed.json  │  ← remembers the latest message timestamp
│                     │     so next run only fetches new messages
└─────────────────────┘
```

> **Tip:** Run `python ingest.py --update` to only fetch new messages instead of rebuilding from scratch.

---

## Live Pipeline — Answering Your Question

Run the chatbot:

```
python aiclient.py
```

### What happens when you type a question

```
You type a question
       │
       ▼
┌──────────────┐
│ aiclient.py  │  ← wraps your question with:
│              │     - system instructions
│              │     - channel name → ID map
│              │     - last 4 exchanges of history
└──────┬───────┘
       │  spawns as subprocess
       ▼
┌──────────────────────────────┐
│  auggie CLI  (Claude Sonnet) │  ← reads the full prompt
│                              │     decides which tool to use
└──────────────┬───────────────┘
               │  communicates over MCP (stdin/stdout pipe)
               ▼
┌─────────────────────────────────────────────────────┐
│                     server.py                       │
│                  (FastMCP server)                   │
│                                                     │
│   Tool 1: read_messages   → Slack API               │
│   Tool 2: send_message    → Slack API               │
│   Tool 3: semantic_search → Qdrant DB               │
└─────────────────────────────────────────────────────┘
               │
               │  tool result sent back to auggie
               ▼
┌──────────────────────────────┐
│  auggie writes final answer  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────┐
│ aiclient.py  │  ← prints the answer under "AI Final Reply"
└──────────────┘
```

---

## Decision Flowchart — How auggie Picks a Tool

```
You ask a question
       │
       ▼
  Is it about recent or live messages?
       │
      YES ──────────────────────────────────► read_messages
       │                                       (calls Slack API directly)
      NO
       │
       ▼
  Are you asking to send a message?
       │
      YES ──────────────────────────────────► send_message
       │                                       (posts to Slack channel)
      NO
       │
       ▼
  Is it about a past discussion, problem, or solution?
       │
      YES ──────────────────────────────────► semantic_search
       │                                       (searches Qdrant vector DB)
      NO
       │
       ▼
  auggie answers from context or says "I don't know"
```

---

## The Three MCP Tools Explained

### Tool 1 — `read_messages`

**Triggered by:** "Show me the last 10 messages in #general" / "What's latest in #infra?"

```
channel_id + limit
       │
       ▼
Slack API → conversations.history
       │
       ├── Has thread replies? → fetch replies and indent them
       │
       └── Has images? → vision.py → auggie → text description
       │
       ▼
Returns formatted list of messages
```

---

### Tool 2 — `send_message`

**Triggered by:** "Send 'Hi team' to #general"

```
channel_id + text
       │
       ▼
Slack API → chat.postMessage
       │
       ▼
Returns success or error status
```

> **Note:** The system prompt tells auggie to always confirm with you before sending.

---

### Tool 3 — `semantic_search`

**Triggered by:** "What was the fix for the Victoria Metrics issue?" / "How did we solve the Grafana permissions problem?"

```
Your query (plain English)
       │
       ▼
embeddings.py converts query → vector
       │
       ▼
Qdrant finds closest matching vectors
       │
       ▼
Results filtered: score must be ≥ 0.55
       │
       ├── Score too low? → "No relevant discussion found. Do not fabricate."
       │
       └── Good match found?
               │
               ▼
       Results labelled:
         🎯 = matched message (the question / topic)
         💬 = thread reply   (the answer / solution)
         ►  = surrounding context
               │
               ▼
       auggie reads labels and writes a grounded answer
```

---

## How MCP Works in This Project

MCP (Model Context Protocol) is the communication layer between `auggie` (the brain) and `server.py` (the hands).

```
aiclient.py
    │
    │  1. writes a temp JSON config file:
    │     { "mcpServers": { "slack": { "command": "python", "args": ["server.py"] } } }
    │
    │  2. spawns auggie with --mcp-config <that file>
    │
    ▼
auggie
    │
    │  3. reads the config → starts server.py as a subprocess
    │
    │  4. communicates over stdin/stdout pipe (MCP protocol)
    │
    ▼
server.py
    │
    │  5. auggie calls a tool (e.g. semantic_search)
    │     → server.py runs the function
    │     → sends the result back over the pipe
    │
    ▼
auggie gets the result → writes final answer → aiclient.py prints it
```

**Why FastMCP?** Just a `@mcp.tool()` decorator above a Python function is enough to expose it as a tool. No web server, no REST API, no schema files needed.

---

## End-to-End Example

> **You:** *what was the fix for the grafana dashboard permissions issue?*

```
Step 1 — aiclient.py builds the prompt and spawns auggie

Step 2 — auggie decides: this is a historical question → calls semantic_search
          query = "grafana dashboard permissions fix"

Step 3 — server.py runs semantic_search:
          → embeddings.py converts query to a vector
          → Qdrant finds the closest matching thread (score: 0.78)
          → score 0.78 ≥ 0.55 → result passes the filter

Step 4 — Result returned to auggie:
          🎯 "Does anyone know how to give the on-call team editor access on Grafana?"
          💬 "Yep — go to Team Settings → grant Editor role to on-call-rotation group"
          💬 "@alice applied the change on 2026-03-14"

Step 5 — auggie reads the 💬 replies and writes:
          "The thread in #infra concluded that the on-call rotation
           should be granted the Editor role in Grafana Team Settings.
           @alice applied the change on 2026-03-14."

Step 6 — aiclient.py prints the answer
```

No hallucination — the answer came entirely from the real indexed thread.
If the top match had scored below `0.55`, auggie would have replied:
*"No relevant discussion found in Slack for this topic."*

---

## Quick Reference — Run Order

```
# First time setup
python fetch_channels.py      # 1. get channel map
python ingest.py              # 2. build knowledge base

# Start chatting
python aiclient.py            # 3. launch the chatbot

# Keep knowledge base fresh
python ingest.py --update     # run periodically for new messages

# Test search without the chatbot
python search_cli.py "your query here"
```

---

## TL;DR

- **`fetch_channels.py`** → saves channel names to a file (run once)
- **`ingest.py`** → fills the memory (run offline)
- **`aiclient.py`** → the prompt you talk to
- **`auggie`** → the AI brain that decides what to do
- **`server.py`** → the hands that actually touch Slack and the database
- **`embeddings.py` + Qdrant** → the searchable memory
- **`vision.py`** → turns images into searchable text
- **MCP** → the communication pipe between auggie and server.py