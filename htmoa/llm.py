"""LLM backends: the Anthropic API and an offline mock for plumbing tests."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import random
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# $ per million tokens (input, output). Used only for the cost estimate.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


RETRY_ATTEMPTS = 9   # per call, on rate limits, connection errors and 5xx

# Models that still accept `temperature` and have no `effort` parameter.
LEGACY_SAMPLING_MODELS = {"claude-haiku-4-5"}


log = logging.getLogger("htmoa")


def load_dotenv(path: Path = Path(__file__).resolve().parent.parent / ".env") -> None:
    """Read KEY=VALUE lines from a local .env file into the environment (existing
    variables win). Lets the API key live in a file instead of the shell session."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip().removeprefix("export ").strip(),
                                  value.strip().strip("\"'"))


class FatalLLMError(RuntimeError):
    """Configuration-level failure (auth, bad request, unknown model): abort the experiment."""


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int   # includes thinking tokens
    latency_s: float
    stop_reason: str


class AnthropicLLM:
    """One Messages API call per node. Only the visible text of a response is
    ever forwarded to another node; thinking stays inside the call."""

    def __init__(self, model: str = "claude-haiku-5-5", max_tokens: int = 8000,
                 effort: str | None = "medium", concurrency: int = 8,
                 temp_exp: float | None = None, temp_agg: float | None = None,
                 thinking: str = "adaptive"):
        # thinking="off" disables hidden reasoning: all reasoning is then visible
        # chain-of-thought text, counted and capped by max_tokens like any other output.
        self.thinking = thinking
        import anthropic
        self._anthropic = anthropic
        load_dotenv()
        self.calls = self.input_tokens = self.output_tokens = self.retries = 0  # running totals
        self.stop_reasons: Counter = Counter()
        try:
            self.client = anthropic.AsyncAnthropic(max_retries=6)
        except Exception as e:  # no credentials
            raise FatalLLMError(f"Could not create the Anthropic client: {e}\n"
                                "Set ANTHROPIC_API_KEY (or run `ant auth login`).") from e
        legacy = model in LEGACY_SAMPLING_MODELS
        if not legacy and (temp_exp is not None or temp_agg is not None):
            raise FatalLLMError(
                f"{model} rejects non-default sampling parameters, so --temp-exp/--temp-agg "
                f"cannot be used with it. Models that accept them: {sorted(LEGACY_SAMPLING_MODELS)}.")
        self.model, self.max_tokens = model, max_tokens
        self.effort = None if legacy else effort   # legacy models have no effort parameter
        # Sampling temperatures (tau_exp / tau_agg in the draft), legacy models only.
        self.temps = {"gen": temp_exp, "agg": temp_agg}
        self._sem = asyncio.Semaphore(concurrency)
        self._native_effort = "output_config" in inspect.signature(
            self.client.messages.create).parameters

    async def complete(self, system: str, user: str, *, phase: str, meta: dict) -> Completion:
        kwargs: dict = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                            messages=[{"role": "user", "content": user}])
        if self.effort:
            if self._native_effort:
                kwargs["output_config"] = {"effort": self.effort}
            else:  # SDK older than the parameter: send it in the body
                kwargs["extra_body"] = {"output_config": {"effort": self.effort}}
        if self.thinking == "off" and self.model not in LEGACY_SAMPLING_MODELS:
            kwargs["thinking"] = {"type": "disabled"}
        if self.temps.get(phase) is not None:
            kwargs["temperature"] = self.temps[phase]
        a = self._anthropic
        async with self._sem:
            t0 = time.perf_counter()
            for attempt in range(RETRY_ATTEMPTS):
                try:
                    # streamed so that a large max_tokens cannot hit an HTTP timeout
                    async with self.client.messages.stream(**kwargs) as stream:
                        resp = await stream.get_final_message()
                    break
                except (a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError,
                        a.BadRequestError) as e:
                    raise FatalLLMError(f"{type(e).__name__}: {e}") from e
                except TypeError as e:  # the SDK raises this when no credentials are found
                    raise FatalLLMError(
                        f"{e}\nSet ANTHROPIC_API_KEY (or run `ant auth login`).") from e
                except (a.RateLimitError, a.APIConnectionError, a.InternalServerError) as e:
                    # Beyond the SDK's own retries: wait while still holding the slot, so a
                    # rate limit slows the whole experiment down instead of failing the run
                    # and wasting the calls it already paid for.
                    if attempt == RETRY_ATTEMPTS - 1:
                        raise
                    self.retries += 1
                    wait = min(90.0, 5.0 * 2 ** attempt) * (0.5 + random.random())
                    log.warning("%s on %s; retry %d/%d in %.0fs", type(e).__name__,
                                meta.get("key", "?"), attempt + 1, RETRY_ATTEMPTS - 1, wait)
                    await asyncio.sleep(wait)
            dt = time.perf_counter() - t0
        u = resp.usage
        text = "".join(b.text for b in resp.content if b.type == "text")
        tin = (u.input_tokens + (getattr(u, "cache_creation_input_tokens", 0) or 0)
               + (getattr(u, "cache_read_input_tokens", 0) or 0))
        stop = resp.stop_reason or ""
        self.calls += 1
        self.input_tokens += tin
        self.output_tokens += u.output_tokens
        self.stop_reasons[stop] += 1
        if stop != "end_turn":
            log.warning("call %s stopped with %r (%d output tokens, %d chars of text)",
                        meta.get("key", "?"), stop, u.output_tokens, len(text))
        log.debug("call %s: %d in, %d out, %.1fs", meta.get("key", "?"), tin, u.output_tokens, dt)
        return Completion(text=text, input_tokens=tin, output_tokens=u.output_tokens,
                          latency_s=dt, stop_reason=stop)

    @property
    def spent_usd(self) -> float:
        """Cost of every completed call so far, including calls of runs that later failed."""
        return cost_usd(self.model, self.input_tokens, self.output_tokens) or 0.0


class MockLLM:
    """Offline stand-in that returns noisy heuristic plans. It exists ONLY to test
    the pipeline end to end without API calls; its scores say nothing about HT-MoA."""

    model = "mock"
    spent_usd = 0.0

    def __init__(self, concurrency: int = 64):
        self._sem = asyncio.Semaphore(concurrency)
        self.calls = 0
        self.stop_reasons: Counter = Counter()

    async def complete(self, system: str, user: str, *, phase: str, meta: dict) -> Completion:
        from .env import verify
        from .solver import anneal, to_canonical
        inst = meta["inst"]
        rng = random.Random(zlib.crc32(meta["key"].encode()))
        async with self._sem:
            await asyncio.sleep(0)
            children = [c for c in meta.get("children", []) if c.plan is not None]
            if phase == "agg" and children:
                ranked = sorted(children, key=lambda c: (verify(inst, c.plan).valid,
                                                         verify(inst, c.plan).profit))
                plan = (ranked[-1] if rng.random() < 0.8 else rng.choice(children)).plan
                plan = json.loads(json.dumps(plan))
            else:
                plan = to_canonical(anneal(inst, rng.randint(150, 900), rng)[0])
            if rng.random() < 0.2 and plan["quarters"][0]["assignments"]:  # inject an error
                plan["quarters"][0]["assignments"][0]["senior"] += 3
            text = "<rationale>mock rationale</rationale>\n<plan>\n" + json.dumps(plan) + "\n</plan>"
            if meta.get("n_branches"):
                text += "\n<branches>\n" + "\n".join(
                    f"{i + 1}. mock commitment {i + 1}" for i in range(meta["n_branches"])) + "\n</branches>"
            if phase == "agg" and meta.get("delta") == 2:
                alt = to_canonical(anneal(inst, rng.randint(150, 900), rng)[0])
                text += "\n<alternative>\n" + json.dumps(alt) + "\n</alternative>"
            text += "\n<dissent>- mock note</dissent>"
        self.calls += 1
        self.stop_reasons["end_turn"] += 1
        return Completion(text=text, input_tokens=(len(system) + len(user)) // 4,
                          output_tokens=len(text) // 4, latency_s=0.0, stop_reason="end_turn")


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    pi, po = PRICES[model]
    return (input_tokens * pi + output_tokens * po) / 1e6
