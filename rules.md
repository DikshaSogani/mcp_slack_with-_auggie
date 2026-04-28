# Slack MCP Agent Rules and Setup

This document consolidates all rules, instructions, and architecture details for the Slack MCP Agent setup.

## 1. System Architecture
- **AI Client (`aiclient.py`)**: Connects to the Groq LLM (LLaMA 3.3 70B Versatile) for tool calling and reasoning, and to the MCP server.
- **MCP Server (`server.py`)**: Uses FastMCP to expose Slack functionality (reading/sending messages, semantic searching) as MCP tools.
- **Semantic Search Engine (`embeddings.py`, `ingest.py`)**: Uses the `sentence-transformers/all-MiniLM-L6-v2` local model and FAISS vector index to implement semantic search across Slack channels locally.
- **Channel Manager (`fetch_channels.py`, `channels.json`)**: Extracts public slack channels and maps names to Slack channel IDs so that users don't have to manually remember IDs.

## 2. Setup Prerequisites
- Python 3.14+ (or compatible version).
- Activate your Python virtual environment.
- Install dependencies from `requirements.txt`.
- `.env` file must be present at the project root with the following keys:
  - `SLACK_BOT_TOKEN`: The Slack app token with required scopes (`channels:history`, `channels:read`, `chat:write`, `groups:history`, and **`files:read`** for image support).
  - `GROQ_API_KEY`: The API key to use Groq LLMs.

## 3. Image Indexing (New)
- **Model**: Uses `meta-llama/llama-4-scout-17b-16e-instruct` (multimodal) on Groq.
- **Process**: During ingestion, if a message contains an image, the system:
    1. Downloads the private image using the bot token.
    2. Sends it to Groq for a detailed textual description.
    3. Indexes the description in the local FAISS semantic index.
- **Search**: You can search for visual details (e.g., "show me the architecture diagram" or "find the screenshot of the bug") just like text.

## 4. General Workflow

1. **Populate Channels:** Run `python fetch_channels.py` to create/update `channels.json` mappings.
2. **Build Knowledge Base:** Run `python ingest.py` periodically offline to fetch channel histories and build the local FAISS semantic index (`slack_faiss_index.index` and `slack_metadata.json`).
3. **Start Agent:** Run `python aiclient.py` to launch the interactive prompt.

## 4. Interaction Rules for the LLM
- The LLM will silently determine and execute function calls (`read_messages`, `send_message`, `semantic_search`) using native tool-call formats rather than text blocks. If multiple steps are required, it resolves them recursively behind the scenes.
- **Semantic Search**: The AI prioritizes `semantic_search` to find relevant past discussions across all channels rather than indiscriminately scrolling a single channel. It ignores unrelated context.
- **Multiple Answers**: If search finds multiple relevant solutions or contexts, the AI MUST mention all of them instead of only the first one.
- **Reading verbatim messages**: When explicitly requested to read/list messages, the AI returns the verbatim text and does not summarize them into a single response.
- **Channel Names**: Using the inline loaded `channels.json` metadata, if the user mentions a channel by Name rather than ID, the agent automatically swaps in the corresponding Slack ID (`C0...`) when issuing the tool call.

## 5. Maintenance
- **Reindexing**: To update the memory of the agent with newer chats, `python ingest.py` must be re-run manually.
- **New Channels**: To allow the agent to know newly created channels, run `python fetch_channels.py`.