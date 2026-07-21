"""Anthropic client helpers: batch submission, polling, token counting.

All LLM traffic goes through the Message Batches API (50% price) except tiny
interactive calls (cluster labels). Model + prompt versions are stamped onto
every enriched record for auditability.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from rich.console import Console

from . import config

console = Console()

MODEL = config.settings()["llm"]["model"]

# ponytail: no ANTHROPIC_API_KEY → run requests through the user's Claude
# subscription via headless `claude -p`. Same request/result interface as the
# Batches API so the agents don't change. Slower, $0, no schema enforcement
# (schema is embedded in the prompt instead; bad JSON → errored request,
# which the agents already tolerate).
CLI_MODE = False
CLI_MODEL = "sonnet"  # bumped from haiku: category discovery/assignment quality is worth the quota
_cli_batches: dict[str, list[Request]] = {}
_cli_results: dict[str, list[tuple[str, Any]]] = {}

# Batches API price = 50% of standard. claude-opus-4-8: $5/$25 per MTok standard.
BATCH_INPUT_PER_MTOK = 2.50
BATCH_OUTPUT_PER_MTOK = 12.50


def client() -> anthropic.Anthropic | None:
    global CLI_MODE, MODEL
    key = config.env("ANTHROPIC_API_KEY")
    if key:
        return anthropic.Anthropic(api_key=key)
    if shutil.which("claude"):
        CLI_MODE = True
        MODEL = f"{CLI_MODEL} (claude-cli subscription)"  # stamped on records for audit
        console.print(
            "[yellow]ANTHROPIC_API_KEY not set — falling back to the Claude "
            "subscription via headless `claude -p` (slower, $0)[/yellow]"
        )
        return None
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set in .env and no `claude` CLI found — "
        "enrichment/insight stages need one of the two."
    )


def count_input_tokens(c: anthropic.Anthropic | None, system: str, user_content: str) -> int:
    if c is None:
        return (len(system) + len(user_content)) // 4  # rough; cost is $0 in CLI mode
    return c.messages.count_tokens(
        model=MODEL,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    ).input_tokens


def estimate_cost(input_tokens: int, est_output_tokens: int) -> float:
    if CLI_MODE:
        return 0.0
    return (
        input_tokens / 1e6 * BATCH_INPUT_PER_MTOK
        + est_output_tokens / 1e6 * BATCH_OUTPUT_PER_MTOK
    )


def _parse_json(text: str) -> Any | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _run_cli_request(req: Request) -> tuple[str, Any]:
    params = req["params"]
    prompt = params["system"] + "\n\n" + params["messages"][0]["content"]
    schema = (params.get("output_config") or {}).get("format", {}).get("schema")
    if schema:
        prompt += (
            "\n\nYour entire output MUST be a single JSON object valid against this "
            f"JSON Schema — no markdown fences, no commentary:\n{json.dumps(schema)}"
        )
    for _ in range(2):  # ponytail: one retry on bad JSON, then give up (agents tolerate None)
        try:
            out = subprocess.run(
                ["claude", "-p", "--model", CLI_MODEL],
                input=prompt, capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            continue
        parsed = _parse_json(out.stdout)
        if parsed is not None:
            return req["custom_id"], parsed
    return req["custom_id"], None


def submit_batch(c: anthropic.Anthropic | None, requests: list[Request]) -> str:
    if c is None:
        batch_id = f"cli-{len(_cli_batches)}"
        _cli_batches[batch_id] = requests
        console.print(f"queued {len(requests)} requests for `claude -p` as {batch_id}")
        return batch_id
    batch = c.messages.batches.create(requests=requests)
    console.print(f"submitted batch {batch.id} with {len(requests)} requests")
    return batch.id


def wait_for_batch(c: anthropic.Anthropic | None, batch_id: str, max_wait_minutes: int = 90,
                   poll_seconds: int = 30) -> None:
    if c is None:
        reqs = _cli_batches[batch_id]
        results: list[tuple[str, Any]] = []
        # ponytail: 30 workers tripped subscription rate limits (60% errors); 10 is the sweet spot
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(_run_cli_request, r) for r in reqs]
            for i, f in enumerate(as_completed(futures), 1):
                results.append(f.result())
                if i % 10 == 0 or i == len(reqs):
                    console.print(f"  {batch_id}: {i}/{len(reqs)} done")
        _cli_results[batch_id] = results
        ok = sum(1 for _, p in results if p is not None)
        console.print(f"batch {batch_id} ended: ok={ok} err={len(results) - ok}")
        return
    start = time.time()
    while True:
        b = c.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            console.print(
                f"batch {batch_id} ended: ok={b.request_counts.succeeded} "
                f"err={b.request_counts.errored} exp={b.request_counts.expired}"
            )
            return
        elapsed = (time.time() - start) / 60
        if elapsed > max_wait_minutes:
            raise TimeoutError(f"batch {batch_id} still {b.processing_status} after {elapsed:.0f}m")
        console.print(
            f"  batch {batch_id}: {b.processing_status}, "
            f"processing={b.request_counts.processing} done={b.request_counts.succeeded} "
            f"({elapsed:.1f}m elapsed)"
        )
        time.sleep(poll_seconds)


def iter_batch_results(c: anthropic.Anthropic | None, batch_id: str) -> Iterable[tuple[str, Any]]:
    """Yield (custom_id, parsed_json_or_None). Errored requests yield None."""
    if c is None:
        yield from _cli_results[batch_id]
        return
    for result in c.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            yield result.custom_id, None
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        parsed = _parse_json(text)
        yield result.custom_id, parsed


def build_request(custom_id: str, system: str, user_content: str, max_tokens: int,
                  schema: dict | None = None) -> Request:
    params: dict[str, Any] = dict(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    if schema is not None:
        params["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    return Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**params))
