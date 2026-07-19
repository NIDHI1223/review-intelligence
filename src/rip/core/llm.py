"""Anthropic client helpers: batch submission, polling, token counting.

All LLM traffic goes through the Message Batches API (50% price) except tiny
interactive calls (cluster labels). Model + prompt versions are stamped onto
every enriched record for auditability.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from rich.console import Console

from . import config

console = Console()

MODEL = config.settings()["llm"]["model"]

# Batches API price = 50% of standard. claude-opus-4-8: $5/$25 per MTok standard.
BATCH_INPUT_PER_MTOK = 2.50
BATCH_OUTPUT_PER_MTOK = 12.50


def client() -> anthropic.Anthropic:
    key = config.env("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set in .env — enrichment/insight stages need it."
        )
    return anthropic.Anthropic(api_key=key)


def count_input_tokens(c: anthropic.Anthropic, system: str, user_content: str) -> int:
    return c.messages.count_tokens(
        model=MODEL,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    ).input_tokens


def estimate_cost(input_tokens: int, est_output_tokens: int) -> float:
    return (
        input_tokens / 1e6 * BATCH_INPUT_PER_MTOK
        + est_output_tokens / 1e6 * BATCH_OUTPUT_PER_MTOK
    )


def submit_batch(c: anthropic.Anthropic, requests: list[Request]) -> str:
    batch = c.messages.batches.create(requests=requests)
    console.print(f"submitted batch {batch.id} with {len(requests)} requests")
    return batch.id


def wait_for_batch(c: anthropic.Anthropic, batch_id: str, max_wait_minutes: int = 90,
                   poll_seconds: int = 30) -> None:
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


def iter_batch_results(c: anthropic.Anthropic, batch_id: str) -> Iterable[tuple[str, Any]]:
    """Yield (custom_id, parsed_json_or_None). Errored requests yield None."""
    for result in c.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            yield result.custom_id, None
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            yield result.custom_id, json.loads(text)
        except json.JSONDecodeError:
            yield result.custom_id, None


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
