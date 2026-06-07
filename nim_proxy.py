"""
nim_proxy.py — Clean Claude Code → NVIDIA NIM proxy
=====================================================
Intercepts Claude Code calls (Anthropic Messages API format)
and translates them to the OpenAI-compatible format used by NVIDIA NIM.

Full flow:
    Claude Code  →  POST /v1/messages (Anthropic SSE)
                 →  This proxy
                 →  POST /v1/chat/completions (OpenAI SSE → NIM)
                 →  Response translated back to Anthropic SSE
                 →  Claude Code

Dependencies: fastapi, uvicorn, httpx, python-dotenv
"""

import asyncio
import json
import os
import random
import time
import uuid
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY", "")
NIM_MODEL   = os.getenv("NIM_MODEL", "moonshotai/kimi-k2.5")
NIM_BASE    = "https://integrate.api.nvidia.com/v1"
PROXY_PORT  = int(os.getenv("PROXY_PORT", "8082"))

if not NIM_API_KEY:
    raise RuntimeError("❌  Missing NVIDIA_NIM_API_KEY in the .env file")

app = FastAPI(title="Claude Code → NVIDIA NIM proxy")


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST CONVERSION: Anthropic → OpenAI
# ══════════════════════════════════════════════════════════════════════════════

def convert_image_block(block: dict) -> dict | None:
    """
    Converts an Anthropic image block to an OpenAI image_url part.

    Anthropic: {"type": "image", "source": {"type": "base64",
                "media_type": "image/png", "data": "..."}}
           or  {"type": "image", "source": {"type": "url", "url": "https://..."}}
    OpenAI:    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    """
    source = block.get("source") or {}
    stype  = source.get("type")
    if stype == "base64":
        media = source.get("media_type", "image/png")
        data  = source.get("data", "")
        return {"type": "image_url",
                "image_url": {"url": f"data:{media};base64,{data}"}}
    if stype == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url", "")}}
    return None


def convert_messages(body: dict) -> list[dict]:
    """
    Converts the Anthropic message array to the OpenAI format.

    Main differences:
    - Anthropic: system is a separate field in the body
    - Anthropic: content can be a string or a list of content blocks
    - Anthropic: tool_result lives in user messages as a content block
    - OpenAI:    tool_result is a message with role="tool"
    - Anthropic: tool_use lives in assistant messages as a content block
    - OpenAI:    tool_use lives in the message's tool_calls field
    """
    messages: list[dict] = []

    # 1. System prompt (Anthropic keeps it separate, OpenAI puts it first)
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # List of content blocks (usually only type "text")
            text = "\n".join(
                b.get("text", "") for b in system if b.get("type") == "text"
            )
            if text:
                messages.append({"role": "system", "content": text})

    # 2. Message history
    for msg in body.get("messages", []):
        role    = msg["role"]
        content = msg["content"]

        # ── User messages ─────────────────────────────────────────────────
        if role == "user":
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})

            elif isinstance(content, list):
                text_blocks   = [b for b in content if b.get("type") == "text"]
                image_blocks  = [b for b in content if b.get("type") == "image"]
                tool_results  = [b for b in content if b.get("type") == "tool_result"]

                # tool_results become separate messages with role="tool"
                for tr in tool_results:
                    tr_content = tr.get("content") or ""  # guard: content may be null
                    if isinstance(tr_content, list):
                        tr_content = "\n".join(
                            b.get("text", "") for b in tr_content
                            if b.get("type") == "text"
                        )
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content":      tr_content,
                    })

                # User text + images. When images are present, OpenAI needs the
                # content as a list of parts (text + image_url) instead of a
                # plain string — otherwise the image is silently dropped.
                if image_blocks:
                    parts: list[dict] = [
                        {"type": "text", "text": b.get("text", "")}
                        for b in text_blocks if b.get("text")
                    ]
                    for b in image_blocks:
                        part = convert_image_block(b)
                        if part:
                            parts.append(part)
                    if parts:
                        messages.append({"role": "user", "content": parts})
                elif text_blocks:
                    text = "\n".join(b.get("text", "") for b in text_blocks)
                    messages.append({"role": "user", "content": text})

        # ── Assistant messages ────────────────────────────────────────────
        elif role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})

            elif isinstance(content, list):
                text_blocks     = [b for b in content if b.get("type") == "text"]
                tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]

                oai_msg: dict = {"role": "assistant", "content": None}

                if text_blocks:
                    oai_msg["content"] = "\n".join(
                        b.get("text", "") for b in text_blocks
                    )

                if tool_use_blocks:
                    oai_msg["tool_calls"] = [
                        {
                            "id":   block["id"],
                            "type": "function",
                            "function": {
                                "name":      block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                        for block in tool_use_blocks
                    ]

                messages.append(oai_msg)

    return messages


def convert_tools(tools: list[dict]) -> list[dict]:
    """
    Converts tool definitions from Anthropic → OpenAI.

    Anthropic uses `input_schema`, OpenAI uses `parameters`.
    """
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t.get("description", ""),
                "parameters":  t.get("input_schema", {}),
            },
        }
        for t in tools
    ]


def build_nim_request(body: dict) -> dict:
    """
    Builds the full payload for the NIM API.
    Only forwards parameters NIM understands; drops the Anthropic-specific ones.
    """
    payload: dict = {
        "model":      NIM_MODEL,
        "max_tokens": body.get("max_tokens", 4096),
        "messages":   convert_messages(body),
        "stream":     body.get("stream", False),
    }

    # OpenAI-compatible APIs (NIM included) only send token usage in the
    # stream when explicitly asked. Without this, Claude Code's cost/context
    # counters stay at 0 during streaming.
    if payload["stream"]:
        payload["stream_options"] = {"include_usage": True}

    # Optional sampling parameters
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]

    # Tools (function calling)
    tools = body.get("tools")
    if tools:
        payload["tools"] = convert_tools(tools)

        # tool_choice conversion
        tc = body.get("tool_choice")
        if tc:
            tc_type = tc.get("type")
            if tc_type == "auto":
                payload["tool_choice"] = "auto"
            elif tc_type == "any":
                payload["tool_choice"] = "required"
            elif tc_type == "tool":
                payload["tool_choice"] = {
                    "type":     "function",
                    "function": {"name": tc.get("name", "")},
                }

    return payload


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING RESPONSE CONVERSION: OpenAI SSE → Anthropic SSE
# ══════════════════════════════════════════════════════════════════════════════

def sse(event: str, data: dict) -> str:
    """Formats an SSE event in the format Claude Code expects."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def map_finish_reason(openai_reason: str | None) -> str:
    """Translates OpenAI's finish_reason to Anthropic's stop_reason."""
    return {
        "stop":        "end_turn",
        "tool_calls":  "tool_use",
        "length":      "max_tokens",
        "content_filter": "stop_sequence",
    }.get(openai_reason or "", "end_turn")


async def stream_response(
    nim_response: httpx.Response,
    message_id:   str,
    model_name:   str,
) -> AsyncGenerator[str, None]:
    """
    Translates NIM's SSE stream (OpenAI format) to Anthropic's SSE format.

    Anthropic expects this sequence of events:
        message_start
        ping
        content_block_start  (one per text or tool_use block)
        content_block_delta  (multiple, one per fragment)
        content_block_stop
        message_delta        (with stop_reason and usage)
        message_stop
    """

    # Stream state
    in_text_block     = False
    text_index        = 0        # index of the text block in Anthropic
    in_thinking_block = False
    thinking_index    = 0        # index of the thinking block in Anthropic
    tool_blocks:    dict[int, dict] = {}  # oai index → {id, name, anthropic_index}
    next_block_idx  = 0          # next Anthropic block index to assign
    finish_reason   = "end_turn"
    input_tokens    = 0
    output_tokens   = 0

    # ── Message start ─────────────────────────────────────────────────────────
    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id":            message_id,
            "type":          "message",
            "role":          "assistant",
            "content":       [],
            "model":         model_name,
            "stop_reason":   None,
            "stop_sequence": None,
            "usage":         {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield sse("ping", {"type": "ping"})

    # ── Read the NIM stream line by line ──────────────────────────────────────
    # MID-STREAM ERRORS: if NIM drops the connection midway (e.g. a late 429),
    # aiter_lines() raises RemoteProtocolError / StreamClosed.
    # We catch it here and cleanly yield the error so it doesn't crash.
    try:
      async for line in nim_response.aiter_lines():
        if not line.startswith("data: "):
            continue

        raw = line[6:].strip()
        if raw == "[DONE]":
            break

        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            continue

        # Extract usage if present in the chunk
        if chunk.get("usage"):
            input_tokens  = chunk["usage"].get("prompt_tokens",     input_tokens)
            output_tokens = chunk["usage"].get("completion_tokens",  output_tokens)

        choices = chunk.get("choices") or []
        if not choices:
            continue

        choice = choices[0]
        delta  = choice.get("delta") or {}  # guard: NIM may send delta:null
        fr     = choice.get("finish_reason")
        if fr:
            finish_reason = map_finish_reason(fr)

        # ── Reasoning delta (thinking models, e.g. kimi-k2-thinking) ────────
        # NIM streams chain-of-thought in a separate `reasoning_content` field.
        # We surface it as Anthropic thinking blocks so the terminal shows live
        # progress instead of looking frozen. (Thinking blocks are dropped on
        # the way back in convert_messages, so no signature handling is needed.)
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if not in_thinking_block:
                in_thinking_block = True
                thinking_index    = next_block_idx
                next_block_idx += 1
                yield sse("content_block_start", {
                    "type":          "content_block_start",
                    "index":         thinking_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                })

            yield sse("content_block_delta", {
                "type":  "content_block_delta",
                "index": thinking_index,
                "delta": {"type": "thinking_delta", "thinking": reasoning},
            })

        # ── Text delta ──────────────────────────────────────────────────────
        text = delta.get("content")
        if text:
            # Close the thinking block once the real answer starts
            if in_thinking_block:
                yield sse("content_block_stop", {
                    "type": "content_block_stop", "index": thinking_index,
                })
                in_thinking_block = False

            if not in_text_block:
                in_text_block = True
                text_index    = next_block_idx
                next_block_idx += 1
                yield sse("content_block_start", {
                    "type":          "content_block_start",
                    "index":         text_index,
                    "content_block": {"type": "text", "text": ""},
                })

            yield sse("content_block_delta", {
                "type":  "content_block_delta",
                "index": text_index,
                "delta": {"type": "text_delta", "text": text},
            })

        # ── Tool call delta ─────────────────────────────────────────────────
        for tc in (delta.get("tool_calls") or []):
            oai_idx = tc.get("index", 0)
            fn      = tc.get("function", {})
            name    = fn.get("name")
            args    = fn.get("arguments", "")
            tc_id   = tc.get("id")

            if oai_idx not in tool_blocks:
                # Close the thinking block if it was open
                if in_thinking_block:
                    yield sse("content_block_stop", {
                        "type": "content_block_stop", "index": thinking_index,
                    })
                    in_thinking_block = False

                # Close the text block if it was open
                if in_text_block:
                    yield sse("content_block_stop", {
                        "type": "content_block_stop", "index": text_index,
                    })
                    in_text_block = False

                anthr_idx = next_block_idx
                next_block_idx += 1

                tool_blocks[oai_idx] = {
                    "id":             tc_id or f"toolu_{uuid.uuid4().hex[:12]}",
                    "name":           name or "",
                    "anthropic_index": anthr_idx,
                }

                yield sse("content_block_start", {
                    "type":  "content_block_start",
                    "index": anthr_idx,
                    "content_block": {
                        "type":  "tool_use",
                        "id":    tool_blocks[oai_idx]["id"],
                        "name":  tool_blocks[oai_idx]["name"],
                        "input": {},
                    },
                })

            if args:
                yield sse("content_block_delta", {
                    "type":  "content_block_delta",
                    "index": tool_blocks[oai_idx]["anthropic_index"],
                    "delta": {"type": "input_json_delta", "partial_json": args},
                })

    except (httpx.RemoteProtocolError, httpx.StreamClosed, httpx.ReadError):
        # Connection dropped mid-stream — notify Claude Code
        yield sse("error", {
            "type": "error",
            "error": {
                "type":    "overloaded_error",
                "message": "NIM closed the connection mid-response "
                           "(possibly a late rate limit). Please retry.",
            },
        })
        return

    # ── Close any open blocks ─────────────────────────────────────────────────
    if in_thinking_block:
        yield sse("content_block_stop", {
            "type": "content_block_stop", "index": thinking_index,
        })

    if in_text_block:
        yield sse("content_block_stop", {
            "type": "content_block_stop", "index": text_index,
        })

    for block in tool_blocks.values():
        yield sse("content_block_stop", {
            "type": "content_block_stop", "index": block["anthropic_index"],
        })

    # ── Message end ───────────────────────────────────────────────────────────
    # input_tokens is included here so Claude Code can track context usage
    # (Anthropic carries it in message_start, but we only learn it from NIM's
    # streamed usage chunk, which arrives at the end).
    yield sse("message_delta", {
        "type":  "message_delta",
        "delta": {"stop_reason": finish_reason, "stop_sequence": None},
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })
    yield sse("message_stop", {"type": "message_stop"})


# ══════════════════════════════════════════════════════════════════════════════
# HTTP ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """
    Claude Code sends a HEAD / on startup to check connectivity.
    We respond 200 so the 404 doesn't show up in the logs.
    """
    return Response(status_code=200)


@app.get("/v1/models")
async def list_models():
    """
    Claude Code queries this endpoint on startup.
    We return the configured NIM model as if it were an Anthropic model.
    """
    return {
        "object": "list",
        "data": [{
            "id":         NIM_MODEL,
            "object":     "model",
            "created":    int(time.time()),
            "owned_by":   "nvidia-nim",
        }],
    }


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    """
    Claude Code uses this to estimate the context size.
    We make a simple estimate (~4 characters per token).
    """
    body = await request.json()
    rough_chars = len(json.dumps(body))
    return {"input_tokens": rough_chars // 4}


@app.post("/v1/messages")
async def messages(request: Request):
    """
    Main route: receives requests from Claude Code and forwards them to NIM.
    Supports both normal and streaming (SSE) responses.
    """
    body        = await request.json()
    nim_payload = build_nim_request(body)
    is_stream   = nim_payload.get("stream", False)

    print(f"[NIM] → {NIM_MODEL}  ({'stream' if is_stream else 'non-stream'}, "
          f"{len(nim_payload['messages'])} msgs)")

    nim_headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "text/event-stream" if is_stream else "application/json",
    }

    timeout = httpx.Timeout(600.0, connect=10.0)  # 10 min — thinking models are slow

    # ── Non-streaming response (with retry on 429) ───────────────────────────
    if not is_stream:
        max_retries = 5
        nim_resp = None
        try:
            for attempt in range(max_retries):
                async with httpx.AsyncClient(timeout=timeout) as client:
                    nim_resp = await client.post(
                        f"{NIM_BASE}/chat/completions",
                        headers=nim_headers,
                        json=nim_payload,
                    )
                if nim_resp.status_code != 429:
                    break
                if attempt == max_retries - 1:
                    break
                retry_after = nim_resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after else min(2 ** (attempt + 1), 60)
                wait += random.uniform(0, wait * 0.1)
                print(f"[NIM] 429 (no-stream) - retry {attempt+1}/{max_retries} in {wait:.1f}s")
                await asyncio.sleep(wait)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            # NIM unreachable (down, no internet, wrong NIM_BASE) — return a
            # clean error instead of letting it bubble up as a 500 stack trace.
            return Response(
                content=json.dumps({
                    "type":  "error",
                    "error": {
                        "type":    "api_error",
                        "message": f"Could not reach NVIDIA NIM at {NIM_BASE}: {e}",
                    },
                }),
                status_code=502,
                media_type="application/json",
            )

        if nim_resp.status_code != 200:
            return Response(
                content=nim_resp.content,
                status_code=nim_resp.status_code,
                media_type="application/json",
            )

        oai    = nim_resp.json()
        choice = oai["choices"][0]
        msg    = choice["message"]

        content_blocks = []
        if msg.get("content"):
            content_blocks.append({"type": "text", "text": msg["content"]})

        for tc in msg.get("tool_calls") or []:
            try:
                tc_input = json.loads(tc["function"].get("arguments", "{}"))
            except json.JSONDecodeError:
                tc_input = {}

            content_blocks.append({
                "type":  "tool_use",
                "id":    tc["id"],
                "name":  tc["function"]["name"],
                "input": tc_input,
            })

        usage = oai.get("usage", {})
        return {
            "id":            f"msg_{uuid.uuid4().hex}",
            "type":          "message",
            "role":          "assistant",
            "content":       content_blocks,
            "model":         body.get("model", NIM_MODEL),
            "stop_reason":   map_finish_reason(choice.get("finish_reason")),
            "stop_sequence": None,
            "usage": {
                "input_tokens":  usage.get("prompt_tokens",    0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

    # ── Streaming response ────────────────────────────────────────────────────
    message_id = f"msg_{uuid.uuid4().hex}"
    model_name = body.get("model", NIM_MODEL)

    async def generate():
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{NIM_BASE}/chat/completions",
                        headers=nim_headers,
                        json=nim_payload,
                    ) as nim_resp:

                        # ── Rate limit: wait and retry transparently ────────
                        if nim_resp.status_code == 429:
                            if attempt == max_retries - 1:
                                yield sse("error", {
                                    "type": "error",
                                    "error": {
                                        "type":    "rate_limit_error",
                                        "message": f"NIM limit after {max_retries} retries. "
                                                   "Wait a minute and try again.",
                                    },
                                })
                                return

                            retry_after = nim_resp.headers.get("retry-after")
                            wait = float(retry_after) if retry_after else min(2 ** (attempt + 1), 60)
                            wait += random.uniform(0, wait * 0.1)
                            print(f"[NIM] 429 rate limit - retry {attempt+1}/{max_retries} in {wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue

                        # ── Invalid key ───────────────────────────────────────
                        if nim_resp.status_code == 401:
                            yield sse("error", {
                                "type": "error",
                                "error": {
                                    "type":    "authentication_error",
                                    "message": "Invalid NVIDIA NIM API key. "
                                               "Check NVIDIA_NIM_API_KEY in your .env.",
                                },
                            })
                            return

                        # ── Other HTTP error ──────────────────────────────────
                        if nim_resp.status_code != 200:
                            error_body = await nim_resp.aread()
                            yield sse("error", {
                                "type": "error",
                                "error": {
                                    "type":    "api_error",
                                    "message": f"NIM error ({nim_resp.status_code}): "
                                               + error_body.decode()[:200],
                                },
                            })
                            return

                        # ── OK: stream the response ───────────────────────────
                        async for chunk in stream_response(nim_resp, message_id, model_name):
                            yield chunk
                        return  # success

            except httpx.ReadTimeout:
                yield sse("error", {
                    "type": "error",
                    "error": {
                        "type":    "overloaded_error",
                        "message": "NIM did not respond within 10 minutes. Try a faster "
                                   "model (e.g. kimi-k2.5) or retry.",
                    },
                })
                return

            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                # NIM unreachable (down, no internet, wrong NIM_BASE)
                yield sse("error", {
                    "type": "error",
                    "error": {
                        "type":    "api_error",
                        "message": f"Could not reach NVIDIA NIM at {NIM_BASE}: {e}",
                    },
                })
                return

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Direct entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print(f"\n✅  NIM proxy ready")
    print(f"   NIM API : {NIM_BASE}")
    print(f"   Model   : {NIM_MODEL}")
    print(f"   Port    : {PROXY_PORT}")
    print(f"\n   Point Claude Code to: http://localhost:{PROXY_PORT}")
    print(f"   Variable: ANTHROPIC_BASE_URL=http://localhost:{PROXY_PORT}\n")

    uvicorn.run(app, host="127.0.0.1", port=PROXY_PORT)
