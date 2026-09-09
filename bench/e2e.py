"""End-to-end tool selection: does the model pick the right tool?

`bench/run.py` measures *retrieval* accuracy — did ToolBroker put the right tool
in the candidate list. This measures the claim that actually matters: given the
tools it was handed, did the **model** call the right one.

Two conditions per query:

* ``full``      — every tool in the catalogue goes in the request
* ``toolbroker`` — only the top ``k`` ToolBroker selected

The interesting number is the gap between them as the catalogue grows, and the
point where ``full`` stops being possible at all: OpenAI caps a request at 128
tools, so at 200 and 1000 tools the baseline is not merely worse, it is
unavailable. That is a result, not an error.

Callers are pluggable and live here rather than in the library — ToolBroker
ships no LLM client and reads no API key.

    uv run python bench/e2e.py --caller lexical            # offline, no key
    uv run python bench/e2e.py --caller ollama             # a real model, still no key
    uv run python bench/e2e.py --caller openai --model gpt-4.1-mini
    uv run python bench/e2e.py --caller anthropic --model claude-sonnet-5

Add ``--out results.md`` to write a chart-ready table.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from toolbroker import ToolBroker
from toolbroker.adapters.raw import OpenAIAdapter, flatten_name
from toolbroker.bench import BenchmarkCase
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.index.enrich import tokenize
from toolbroker.sources import StaticJSONSource
from toolbroker.types import Tool

DATA = Path(__file__).parent / "data"
SIZES = (50, 200, 1000)
DEFAULT_K = 5

# Providers reject requests above their own ceiling. Recording the ceiling lets
# the harness report "not runnable" instead of a wall of API errors.
PROVIDER_TOOL_LIMITS = {"openai": 128, "anthropic": None, "lexical": None, "ollama": None}


@dataclass(slots=True)
class CallOutcome:
    """What one model call produced."""

    chosen: str | None
    prompt_tokens: int | None = None
    latency_ms: float = 0.0
    error: str | None = None


class ToolCaller(Protocol):
    """Something that picks at most one tool for a query."""

    @property
    def name(self) -> str:
        """Identifier for the results table."""
        ...

    @property
    def max_tools(self) -> int | None:
        """Provider ceiling on tools per request, or ``None`` for no limit."""
        ...

    def call(self, query: str, tools: Sequence[dict[str, Any]]) -> CallOutcome:
        """Return the flattened tool name the model chose, or ``None``."""
        ...


class LexicalCaller:
    """A deterministic offline stand-in for a model.

    It picks the tool whose description shares the most terms with the query.
    It is not a language model and its absolute accuracy means nothing. It
    exists so the harness itself is testable and runnable in CI without a key,
    and because it degrades with catalogue size for the same structural reason
    a real model does — more distractors, more ways to be wrong.
    """

    name = "lexical (offline)"
    max_tools = None

    def call(self, query: str, tools: Sequence[dict[str, Any]]) -> CallOutcome:
        """Score every tool by term overlap and return the best."""
        started = time.perf_counter()
        terms = set(tokenize(query))
        best_name: str | None = None
        best_score = 0.0

        for entry in tools:
            function = entry.get("function", entry)
            text = f"{function.get('name', '')} {function.get('description', '')}"
            candidate_terms = set(tokenize(text))
            if not candidate_terms:
                continue
            overlap = len(terms & candidate_terms)
            if not overlap:
                continue
            # Normalise by candidate length so a verbose tool does not win on
            # sheer word count.
            score = overlap / (len(candidate_terms) ** 0.5)
            if score > best_score:
                best_score, best_name = score, function.get("name")

        return CallOutcome(
            chosen=best_name,
            prompt_tokens=estimate_tokens(tools),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class OpenAICaller:
    """Real OpenAI function-calling.

    Requires ``pip install openai`` and ``OPENAI_API_KEY``.
    """

    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        """Create a client for ``model``."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("pip install openai to use --caller openai") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("set OPENAI_API_KEY to use --caller openai")
        self._client = OpenAI()
        self._model = model

    @property
    def name(self) -> str:
        """Model identifier."""
        return f"openai:{self._model}"

    @property
    def max_tools(self) -> int | None:
        """OpenAI rejects requests carrying more than 128 tools."""
        return PROVIDER_TOOL_LIMITS["openai"]

    def call(self, query: str, tools: Sequence[dict[str, Any]]) -> CallOutcome:
        """Ask the model to pick a tool, returning whichever it called."""
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                tools=list(tools),
                tool_choice="auto",
                temperature=0,
            )
        except Exception as exc:
            return CallOutcome(
                chosen=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

        latency = (time.perf_counter() - started) * 1000
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None) or []
        usage = getattr(response, "usage", None)
        return CallOutcome(
            chosen=calls[0].function.name if calls else None,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            latency_ms=latency,
        )


class AnthropicCaller:
    """Real Anthropic tool use.

    Requires ``pip install anthropic`` and a credential. Renders through the
    Anthropic schema shape rather than the OpenAI one.

    Credentials are left to the SDK rather than checked here. An unset
    ``ANTHROPIC_API_KEY`` does not mean there is no credential: the SDK also
    resolves ``ANTHROPIC_AUTH_TOKEN`` and an ``ant auth login`` profile, and
    rejecting those with "set ANTHROPIC_API_KEY" would send someone hunting for
    a key they do not need.
    """

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        """Create a client for ``model``."""
        try:
            import anthropic
        except ImportError as exc:
            raise SystemExit("pip install anthropic to use --caller anthropic") from exc
        try:
            self._client = anthropic.Anthropic()
        except Exception as exc:
            raise SystemExit(
                f"no Anthropic credential found ({exc}). Set ANTHROPIC_API_KEY, "
                "or run `ant auth login`."
            ) from exc
        self._model = model

    @property
    def name(self) -> str:
        """Model identifier."""
        return f"anthropic:{self._model}"

    @property
    def max_tools(self) -> int | None:
        """No hard count limit; the request token budget binds first."""
        return PROVIDER_TOOL_LIMITS["anthropic"]

    def call(self, query: str, tools: Sequence[dict[str, Any]]) -> CallOutcome:
        """Ask the model to pick a tool, returning whichever it used."""
        started = time.perf_counter()
        anthropic_tools = [
            {
                "name": entry["function"]["name"],
                "description": entry["function"]["description"],
                "input_schema": entry["function"]["parameters"],
            }
            for entry in tools
        ]
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
                tools=anthropic_tools,
            )
        except Exception as exc:
            return CallOutcome(
                chosen=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

        latency = (time.perf_counter() - started) * 1000
        used = [block for block in response.content if getattr(block, "type", "") == "tool_use"]
        usage = getattr(response, "usage", None)
        return CallOutcome(
            chosen=used[0].name if used else None,
            prompt_tokens=getattr(usage, "input_tokens", None),
            latency_ms=latency,
        )


SYSTEM_PROMPT = (
    "You are connected to a set of tools. Call exactly the one tool that best serves "
    "the user's request. If no tool fits, do not call any tool."
)


def estimate_tokens(tools: Sequence[dict[str, Any]]) -> int:
    """Rough prompt-token cost of a tool block.

    Four characters per token is the usual English approximation. Used only
    when the provider does not report real usage; real numbers always win.
    """
    return len(json.dumps(list(tools))) // 4


@dataclass(slots=True)
class ConditionResult:
    """Outcome of one (caller, catalogue size, condition) combination."""

    caller: str
    size: int
    condition: str
    runnable: bool = True
    skip_reason: str = ""
    correct: int = 0
    wrong: int = 0
    abstained: int = 0
    errors: int = 0
    # Cases where no gold tool reached the model at all. Only retrieval can
    # cause this, so it is the share of the loss that is ToolBroker's fault.
    retrieval_misses: int = 0
    prompt_tokens: list[int] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        """Cases actually sent to the model."""
        return self.correct + self.wrong + self.abstained

    @property
    def accuracy(self) -> float:
        """Share of cases where the model called the expected tool."""
        return self.correct / self.attempted if self.attempted else 0.0

    @property
    def accuracy_given_retrieved(self) -> float:
        """Accuracy over the cases where a gold tool actually reached the model.

        This separates the two ways the pipeline fails. Below it, retrieval
        never shortlisted the tool and the caller had nothing to pick. Above
        it, the shortlist was right and the model still chose wrong. Only the
        first is a retrieval problem, and the headline accuracy hides which.
        """
        reachable = self.attempted - self.retrieval_misses
        return self.correct / reachable if reachable else 0.0

    @property
    def mean_prompt_tokens(self) -> float:
        """Mean tool-block size sent per request."""
        return sum(self.prompt_tokens) / len(self.prompt_tokens) if self.prompt_tokens else 0.0

    @property
    def mean_latency_ms(self) -> float:
        """Mean end-to-end call latency."""
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0


def load_catalogue(size: int, embedder: Any) -> tuple[ToolBroker, list[BenchmarkCase]]:
    """Build an indexed catalogue and its labelled query set."""
    broker = ToolBroker(embedder=embedder, cache_embeddings=True)
    broker.add_source(StaticJSONSource(DATA / f"tools_{size}.json"))
    broker.index()
    cases = [
        BenchmarkCase.model_validate(json.loads(line))
        for line in (DATA / f"queries_{size}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return broker, cases


def _failure(
    query: str,
    expected: object,
    got: str,
    reason: str,
    condition: str,
    size: int,
) -> dict[str, Any]:
    """Record one failure with enough context to read it out of order."""
    return {
        "query": query,
        "expected": expected,
        "got": got,
        "reason": reason,
        "condition": condition,
        "size": size,
    }


def evaluate(
    caller: ToolCaller,
    broker: ToolBroker,
    cases: Sequence[BenchmarkCase],
    *,
    condition: str,
    size: int,
    k: int,
) -> ConditionResult:
    """Run every case through ``caller`` under one condition."""
    result = ConditionResult(caller=caller.name, size=size, condition=condition)
    all_tools: list[Tool] = broker.tools()
    # Gold labels are catalogue ids; the caller only ever sees flattened wire
    # names. Map between them so a miss reports the same name shape as a hit.
    wire_names = {tool.id: flatten_name(tool) for tool in all_tools}

    if condition == "full" and caller.max_tools is not None and len(all_tools) > caller.max_tools:
        result.runnable = False
        result.skip_reason = (
            f"{len(all_tools)} tools exceeds the provider limit of {caller.max_tools}"
        )
        return result

    for case in cases:
        adapter = OpenAIAdapter()
        tools = all_tools if condition == "full" else list(broker.select(case.query, k=k).tools)

        rendered = adapter.render(tools)
        expected = {flatten_name(tool) for tool in tools if tool.id in set(case.expected)}
        outcome = caller.call(case.query, rendered)

        if outcome.error:
            result.errors += 1
            continue
        if outcome.prompt_tokens is not None:
            result.prompt_tokens.append(outcome.prompt_tokens)
        result.latencies.append(outcome.latency_ms)

        if case.is_negative:
            # The right move is to call nothing.
            if outcome.chosen is None:
                result.correct += 1
            else:
                result.wrong += 1
                result.failures.append(
                    _failure(
                        case.query,
                        "(no tool)",
                        outcome.chosen,
                        "called on a negative case",
                        condition,
                        size,
                    )
                )
            continue

        # An empty `expected` means no gold tool survived into the rendered
        # set, so the caller could not have answered correctly however well it
        # read. Reporting that as `expected []` hid the one distinction this
        # benchmark exists to draw, so name it and count it separately.
        missed = not expected
        if missed:
            result.retrieval_misses += 1
        shown = sorted(expected) or sorted(wire_names.get(i, i) for i in case.expected)

        if outcome.chosen is None:
            result.abstained += 1
            result.failures.append(
                _failure(
                    case.query,
                    shown,
                    "(no tool)",
                    "retrieval miss" if missed else "caller abstained",
                    condition,
                    size,
                )
            )
        elif outcome.chosen in expected:
            result.correct += 1
        else:
            result.wrong += 1
            result.failures.append(
                _failure(
                    case.query,
                    shown,
                    outcome.chosen,
                    "retrieval miss" if missed else "caller chose wrong",
                    condition,
                    size,
                )
            )

    return result


class OllamaCaller:
    """A real model, run locally, with no API key and no bill.

    This exists because the provider callers gate the most important numbers in
    this repo behind somebody's credit card. An open model served by Ollama
    answers the same question — does a model choose better from five tools than
    from five hundred — for the cost of the electricity. That also makes the
    result reproducible by a reader who does not trust it, which a benchmark
    published by the library's own author needs more than most.

    Requires a running ``ollama serve`` and a tool-calling model pulled, e.g.
    ``ollama pull qwen3:8b``.
    """

    def __init__(self, model: str = "qwen3:8b", host: str | None = None) -> None:
        """Point the caller at a local Ollama server."""
        import urllib.request

        self._model = model
        self._host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        try:
            with urllib.request.urlopen(f"{self._host}/api/tags", timeout=5):
                pass
        except Exception as exc:
            raise SystemExit(
                f"no Ollama server at {self._host} ({exc}). Start one with `ollama serve`."
            ) from exc

    @property
    def name(self) -> str:
        """Model identifier."""
        return f"ollama:{self._model}"

    @property
    def max_tools(self) -> int | None:
        """No hard count limit; the context window binds first."""
        return PROVIDER_TOOL_LIMITS["ollama"]

    def call(self, query: str, tools: Sequence[dict[str, Any]]) -> CallOutcome:
        """Ask the local model to pick a tool, returning whichever it called."""
        import urllib.request

        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": query}],
                "tools": list(tools),
                "stream": False,
                # Thinking off, temperature zero: this measures tool choice, not
                # reasoning style, and a benchmark that moves between identical
                # runs cannot support the claims made from it.
                "think": False,
                "options": {"temperature": 0, "num_ctx": 40960},
            }
        ).encode()

        started = time.perf_counter()
        try:
            request = urllib.request.Request(
                f"{self._host}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=900) as response:
                body = json.loads(response.read())
        except Exception as exc:
            return CallOutcome(
                chosen=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

        calls = (body.get("message") or {}).get("tool_calls") or []
        return CallOutcome(
            chosen=calls[0]["function"]["name"] if calls else None,
            prompt_tokens=body.get("prompt_eval_count"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def build_caller(kind: str, model: str | None) -> ToolCaller:
    """Construct the requested caller."""
    if kind == "lexical":
        return LexicalCaller()
    if kind == "openai":
        return OpenAICaller(model or "gpt-4.1-mini")
    if kind == "anthropic":
        return AnthropicCaller(model or "claude-sonnet-5")
    if kind == "ollama":
        return OllamaCaller(model or "qwen3:8b")
    raise SystemExit(f"unknown caller {kind!r}")


def render_table(results: Sequence[ConditionResult], k: int) -> str:
    """Render the results as a markdown table."""
    lines = [
        "| tools | condition | accuracy | correct | wrong | no-call | retrieval miss "
        "| accuracy if retrieved | tool tokens | ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        if not row.runnable:
            lines.append(
                f"| {row.size} | {row.condition} | **not runnable** | — | — | — | — | — | — | — |"
            )
            continue
        label = row.condition if row.condition == "full" else f"toolbroker k={k}"
        lines.append(
            f"| {row.size} | {label} | {row.accuracy:.3f} | {row.correct} | {row.wrong} "
            f"| {row.abstained} | {row.retrieval_misses} "
            f"| {row.accuracy_given_retrieved:.3f} "
            f"| {row.mean_prompt_tokens:,.0f} | {row.mean_latency_ms:.0f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run the end-to-end comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--caller", default="lexical", choices=["lexical", "openai", "anthropic", "ollama"]
    )
    parser.add_argument("--model", default=None, help="provider model id")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help="tools ToolBroker selects")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--out", type=Path, help="write a markdown report here")
    args = parser.parse_args()

    if not (DATA / f"tools_{args.sizes[0]}.json").exists():
        raise SystemExit("run `python bench/generate.py` first")

    try:
        from toolbroker.index.embedders.fastembed import FastEmbedEmbedder

        embedder: Any = FastEmbedEmbedder()
    except Exception as exc:
        print(f"fastembed unavailable ({exc}); falling back to the lexical embedder\n")
        embedder = HashingEmbedder(dim=512)

    caller = build_caller(args.caller, args.model)
    results: list[ConditionResult] = []

    for size in args.sizes:
        broker, cases = load_catalogue(size, embedder)
        for condition in ("full", "toolbroker"):
            row = evaluate(caller, broker, cases, condition=condition, size=size, k=args.k)
            results.append(row)
            status = "skipped" if not row.runnable else f"accuracy {row.accuracy:.3f}"
            print(f"{size:>5} tools  {condition:<10} {status}")

    table = render_table(results, args.k)
    print()
    print(f"caller: {caller.name}   embedder: {getattr(embedder, 'id', embedder)}")
    print(table)

    if args.out:
        failures = [
            f"- {failure['size']} tools, {failure['condition']}: `{failure['query']}` → "
            f"expected {failure['expected']}, got `{failure['got']}` "
            f"({failure['reason']})"
            for row in results
            for failure in row.failures[:4]
        ]
        args.out.write_text(
            f"# End-to-end tool selection\n\n"
            f"caller: `{caller.name}`  \nembedder: `{getattr(embedder, 'id', embedder)}`  \n"
            f"k: {args.k}  \ncases: {len(cases)}\n\n{table}\n\n"
            "## Sample failures\n\n"
            "A *retrieval miss* means no correct tool reached the model; anything\n"
            "else means it had the right tool in front of it and chose otherwise.\n\n"
            + "\n".join(failures)
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
