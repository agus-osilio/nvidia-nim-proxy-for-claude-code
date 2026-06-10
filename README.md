# NVIDIA NIM Proxy for Claude Code

A minimal, auditable proxy that lets you run **Claude Code** against the free
**NVIDIA NIM** API instead of Anthropic's paid models. No unnecessary
dependencies, no hidden code: everything it does lives in `nim_proxy.py`
(~720 commented lines).

NVIDIA NIM offers **40 free requests/minute** with high-end open coding models
(Kimi K2, Devstral, GLM, MiniMax, etc.) — **no credit card required**.

> [!IMPORTANT]
> NVIDIA's free tier is intended for development, research, and testing. Before
> using it, take a moment to review the
> [NIM FAQ](https://docs.api.nvidia.com/nim/docs/product) and NVIDIA's
> [Technology Access Terms of Use](https://developer.nvidia.com/legal/terms) to
> make sure your use case is covered.

---

## Table of contents

- [How it works](#how-it-works)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Changing the model](#changing-the-model)
- [Project files](#project-files)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

---

## How it works

Claude Code talks the **Anthropic Messages API**. NVIDIA NIM speaks the
**OpenAI-compatible** Chat Completions API. This proxy sits in the middle and
translates both ways, in real time:

```
Claude Code
   │  POST /v1/messages          (Anthropic format, SSE streaming)
   ▼
nim_proxy  ──►  converts request: Anthropic → OpenAI
   │            POST /v1/chat/completions   (NVIDIA NIM)
   ▼
nim_proxy  ◄──  converts response: OpenAI SSE → Anthropic SSE
   │
   ▼
Claude Code   (sees a normal Anthropic response)
```

Claude Code never knows it isn't talking to Anthropic. The proxy handles the
full conversation: system prompts, tool calling, images, streaming, and token
accounting.

---

## Features

- **Full streaming (SSE)** — responses appear token by token, like the real thing.
- **Tool calling** — Claude Code's tools (file edits, bash, etc.) work end to end.
- **Images** — pasted or dragged images are forwarded to multimodal models (e.g. `kimi-k2.5`).
- **Thinking models** — reasoning from models like `kimi-k2-thinking` is streamed
  live as a collapsible thinking block, so the terminal shows progress instead
  of looking frozen.
- **Token & cost tracking** — real token usage is reported back, so Claude Code's
  context bar and `/cost` work.
- **Automatic retries** — transparent exponential backoff on rate limits (HTTP 429).
- **Clean error messages** — invalid API key, unreachable server, or timeouts
  surface as readable messages instead of crashes (see [Troubleshooting](#troubleshooting)).

---

## Requirements

- **Python 3.10** or higher
- **Claude Code** installed:
  ```bash
  npm install -g @anthropic-ai/claude-code
  ```
- A **free NIM API key**: https://build.nvidia.com/settings/api-keys

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/agus-osilio/nvidia-nim-proxy-for-claude-code.git
cd nvidia-nim-proxy-for-claude-code

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies (only 4)
pip install -r requirements.txt

# 4. Create your .env from the template
cp .env.example .env             # Windows: copy .env.example .env
```

Then open `.env` and add your NIM API key — see [Configuration](#configuration) below.

---

## Configuration

All configuration lives in the `.env` file:

| Variable             | Required | Default                  | Description                          |
|----------------------|----------|--------------------------|--------------------------------------|
| `NVIDIA_NIM_API_KEY` | ✅ Yes   | —                        | Your free NIM key (`nvapi-…`)        |
| `NIM_MODEL`          | No       | `moonshotai/kimi-k2.5`   | Which NIM model to route requests to |
| `PROXY_PORT`         | No       | `8082`                   | Local port the proxy listens on      |

Example `.env`:

```env
NVIDIA_NIM_API_KEY="nvapi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
NIM_MODEL="moonshotai/kimi-k2.5"
PROXY_PORT=8082
```

---

## Usage

You need **two terminals**: one running the proxy, one running Claude Code.

### Step 1 — Start the proxy

```bash
python nim_proxy.py
```

You should see:

```
✅  NIM proxy ready
   NIM API : https://integrate.api.nvidia.com/v1
   Model   : moonshotai/kimi-k2.5
   Port    : 8082

   Point Claude Code to: http://localhost:8082
```

Leave this terminal running.

### Step 2 — Point Claude Code at the proxy

In a **second terminal**, set two environment variables and launch `claude`.
These tell Claude Code to send its requests to the local proxy instead of
Anthropic.

**Windows (PowerShell):**

```powershell
$env:ANTHROPIC_BASE_URL="http://localhost:8082"
$env:ANTHROPIC_AUTH_TOKEN="nim-proxy"
claude
```

**Linux / macOS:**

```bash
export ANTHROPIC_BASE_URL="http://localhost:8082"
export ANTHROPIC_AUTH_TOKEN="nim-proxy"
claude
```

> **Note:** `ANTHROPIC_AUTH_TOKEN` can be any value — the proxy ignores it and
> authenticates to NIM with the key from your `.env`. It only needs to be set so
> Claude Code doesn't complain about a missing token.

That's it. Use Claude Code normally; every request is routed to NVIDIA NIM.

> **Tip:** the environment variables only apply to the terminal session where you
> set them. Open a new terminal and you'll need to set them again (or add them to
> your shell profile / PowerShell `$PROFILE` to make them permanent).

---

## Changing the model

Edit `NIM_MODEL` in your `.env` and restart the proxy:

```env
NIM_MODEL="mistralai/devstral-2-123b"
```

Recommended models for coding:

| Model                          | Context | Highlight             |
|--------------------------------|---------|-----------------------|
| `moonshotai/kimi-k2.5`         | 256K    | Fast, multimodal      |
| `moonshotai/kimi-k2-thinking`  | 131K    | Advanced reasoning    |
| `mistralai/devstral-2-123b`    | 128K    | Best for code         |
| `minimaxai/minimax-m2.5`       | 1M      | Huge context          |
| `z-ai/glm4.7`                  | 128K    | Balanced              |

Full catalog: https://build.nvidia.com/models

---

## Project files

```
nim_proxy/
├── nim_proxy.py      # All the proxy code (~720 lines, fully commented)
├── .env.example      # Configuration template
├── requirements.txt  # Dependencies: fastapi, uvicorn, httpx, python-dotenv
└── README.md         # This file
```

---

## Troubleshooting

| Symptom                                                   | Cause & fix                                                                                              |
|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `Missing NVIDIA_NIM_API_KEY in the .env file` at startup  | You didn't create `.env` or didn't set the key. Copy `.env.example` to `.env` and add your `nvapi-…` key.|
| `Invalid NVIDIA NIM API key`                              | The key in `.env` is wrong or expired. Get a new one at build.nvidia.com.                                |
| `Could not reach NVIDIA NIM at …`                         | No internet, NIM is down, or a network/firewall block. Check your connection and retry.                  |
| `NIM error (404 …)` / model not found                     | `NIM_MODEL` is misspelled or unavailable. Check the spelling against the [model catalog](https://build.nvidia.com/models). |
| `NIM limit after 5 retries` / frequent 429s               | You hit the 40 req/min free-tier limit. Wait a minute, or slow down.                                     |
| Claude Code still uses Anthropic                          | `ANTHROPIC_BASE_URL` isn't set in the same terminal where you ran `claude`. Re-set it and relaunch.      |
| Responses are very slow                                   | Thinking models (`kimi-k2-thinking`) reason before answering. Switch to a faster model like `kimi-k2.5`. |

The proxy prints one line per request (e.g. `[NIM] → moonshotai/kimi-k2.5 (stream, 12 msgs)`),
which is handy for confirming traffic is flowing and which model is being used.

---

## Known limitations

- **Rate limit**: 40 requests/min on the free NIM tier.
- **Thinking models**: `kimi-k2-thinking` can be slower (it generates internal
  reasoning before responding). The reasoning is streamed live as a thinking
  block, so the terminal shows progress instead of looking frozen.
- **Images**: Supported with multimodal models (e.g. `kimi-k2.5`). Images inside
  tool results are not forwarded (the OpenAI tool-message format only accepts text).
- **Tool calling**: Requires a model that supports function calling. All the
  models in the [Changing the model](#changing-the-model) table support it.
- **Token counting**: The `count_tokens` endpoint returns a rough estimate
  (~4 characters per token), not an exact tokenizer count.
- **Open models & usage terms**: NIM serves open-source models, not Anthropic's
  Claude, so responses may differ on complex tasks. The free tier is intended
  for development, research, and testing — if your use case is different, it's
  worth reviewing the [NIM FAQ](https://docs.api.nvidia.com/nim/docs/product)
  (development vs. production) and NVIDIA's
  [Technology Access Terms of Use](https://developer.nvidia.com/legal/terms).
