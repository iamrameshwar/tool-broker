"""``toolbroker`` command line.

Six verbs, matching the things you do with a catalogue:

* ``index``      — discover tools from a config and report what was found
* ``query``      — ask what an agent would get for a prompt, with the full trace
* ``bench``      — measure selection accuracy against a labelled query set
* ``calibrate``  — find a score floor so the catalogue can say "no tool fits"
* ``doctor``     — find tools retrieval can never surface
* ``diff``       — what a policy change grants, for a reviewer
* ``serve``      — expose the catalogue as an MCP server

``query`` is the important one. Being able to type a prompt and see exactly
which tools an agent would receive, with scores and rule firings, is how
someone debugs a bad selection without instrumenting their agent.

Built on argparse so the core CLI adds no dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import __version__
from ..catalog import ToolBroker
from ..errors import ToolBrokerError
from ..observability import configure_logging

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="toolbroker",
        description="Retrieval and policy for agent tool catalogues.",
    )
    parser.add_argument("--version", action="version", version=f"toolbroker {__version__}")
    parser.add_argument("-c", "--config", type=Path, help="path to a toolbroker YAML/JSON config")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--json-logs", action="store_true", help="emit logs as JSON")

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="discover and index tools")
    index_parser.add_argument("--out", type=Path, help="write discovered tools to a JSON file")
    index_parser.add_argument("--json", action="store_true", help="print the report as JSON")

    query_parser = subparsers.add_parser("query", help="show what an agent would receive")
    query_parser.add_argument("text", help="the query to run")
    query_parser.add_argument("-k", type=int, default=5, help="how many tools to return")
    query_parser.add_argument("--agent", help="agent identity, selects the policy")
    query_parser.add_argument("--scope", action="append", default=[], help="a held scope")
    query_parser.add_argument("--namespace", action="append", help="restrict to a namespace")
    query_parser.add_argument("--tag", action="append", help="restrict to a tag")
    query_parser.add_argument(
        "--render", help="also render through an adapter, e.g. openai or anthropic"
    )
    query_parser.add_argument("--json", action="store_true", help="print the selection as JSON")

    bench_parser = subparsers.add_parser("bench", help="score selection against labelled queries")
    bench_parser.add_argument("queries", type=Path, help="JSONL of {query, expected:[tool_id]}")
    bench_parser.add_argument("-k", type=int, default=5, help="cut-off for recall@k")
    bench_parser.add_argument("--agent", help="agent identity to evaluate under")
    bench_parser.add_argument("--json", action="store_true", help="print results as JSON")

    calibrate_parser = subparsers.add_parser(
        "calibrate", help="find a MinScore floor for this catalogue"
    )
    calibrate_parser.add_argument(
        "--samples",
        type=Path,
        help="file of queries you expect to succeed, one per line; without it the "
        "report cannot show what a floor would cost you",
    )
    calibrate_parser.add_argument(
        "--noise",
        type=Path,
        help="file of queries that should match nothing, one per line (defaults to a built-in set)",
    )
    calibrate_parser.add_argument("-k", type=int, default=5, help="tools per selection")
    calibrate_parser.add_argument("--json", action="store_true", help="print as JSON")

    doctor_parser = subparsers.add_parser("doctor", help="find tools retrieval cannot surface")
    doctor_parser.add_argument(
        "--samples",
        type=Path,
        help="file of real queries, one per line; adds which tools none of them reach",
    )
    doctor_parser.add_argument("-k", type=int, default=5, help="selection width to judge against")
    doctor_parser.add_argument(
        "--margin",
        type=float,
        help="flag pairs closer than this as twins; omit to rank them instead, "
        "which is the honest default because the right value depends on the embedder",
    )
    doctor_parser.add_argument(
        "--pairs", type=int, default=20, help="how many of the closest pairs to show"
    )
    doctor_parser.add_argument("--json", action="store_true", help="print as JSON")

    diff_parser = subparsers.add_parser(
        "diff", help="show what a policy change grants, against this catalogue"
    )
    diff_parser.add_argument("before", type=Path, help="the config as it is now")
    diff_parser.add_argument("after", type=Path, help="the config as proposed")
    diff_parser.add_argument(
        "--fail-on-grant",
        action="store_true",
        help="exit non-zero if any agent gains a tool; the CI gate",
    )
    diff_parser.add_argument(
        "--fail-on-high-risk",
        action="store_true",
        help="exit non-zero only if an agent gains a HIGH risk tool",
    )
    diff_parser.add_argument("--json", action="store_true", help="print as JSON")

    serve_parser = subparsers.add_parser("serve", help="run the MCP proxy server")
    serve_parser.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio", help="MCP transport"
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="bind host for http")
    serve_parser.add_argument("--port", type=int, default=8000, help="bind port for http")
    serve_parser.add_argument("-k", type=int, default=5, help="tools returned per search")
    serve_parser.add_argument(
        "--refresh",
        type=float,
        metavar="SECONDS",
        help="re-discover tools on this interval; without it the proxy serves "
        "whatever existed at startup, forever",
    )

    return parser


def _load_broker(args: argparse.Namespace) -> ToolBroker:
    """Build a catalogue from ``--config``, or fail with an actionable message."""
    if args.config is None:
        raise ToolBrokerError(
            "no config supplied. Point at one with: toolbroker -c toolbroker.yaml <command>"
        )
    from ..config import load

    return load(args.config)


def _cmd_index(args: argparse.Namespace, out: Any) -> int:
    """Run discovery and indexing."""
    broker = _load_broker(args)
    tools = broker.discover()
    report = broker.index(tools)

    if args.out:
        payload = [tool.model_dump(mode="json") for tool in tools]
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps({**broker.stats(), "indexed": report.indexed}, indent=2), file=out)
        return EXIT_OK

    print(f"indexed {report.indexed} tools from {len(broker.sources)} source(s)", file=out)
    if report.skipped:
        print(f"skipped {report.skipped}", file=out)
    if report.duplicates:
        print(f"duplicate ids collapsed: {', '.join(report.duplicates)}", file=out)
    stats = broker.stats()
    for namespace, count in stats["namespaces"].items():
        print(f"  {namespace}: {count}", file=out)
    return EXIT_OK


def _cmd_query(args: argparse.Namespace, out: Any) -> int:
    """Run a query and print the selection trace."""
    broker = _load_broker(args)
    broker.index()
    selection = broker.select(
        args.text,
        k=args.k,
        agent=args.agent,
        scopes=args.scope,
        namespaces=args.namespace,
        tags=args.tag,
    )

    if args.json:
        print(json.dumps(selection.model_dump(mode="json"), indent=2, default=str), file=out)
        return EXIT_OK

    print(selection.explain(), file=out)
    if args.render:
        print(file=out)
        print(f"--- rendered for {args.render} ---", file=out)
        print(json.dumps(broker.render(args.render, selection), indent=2, default=str), file=out)
    return EXIT_OK


def _cmd_bench(args: argparse.Namespace, out: Any) -> int:
    """Score the catalogue against a labelled query set."""
    from ..bench import BenchmarkCase, run_benchmark

    broker = _load_broker(args)
    broker.index()

    cases = [
        BenchmarkCase.model_validate(json.loads(line))
        for line in args.queries.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = run_benchmark(broker, cases, k=args.k, agent=args.agent)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2), file=out)
        return EXIT_OK
    print(result.summary(), file=out)
    return EXIT_OK


def _read_lines(path: Path | None) -> list[str] | None:
    """Read a newline-separated query file, ignoring blanks and comments."""
    if path is None:
        return None
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _cmd_calibrate(args: argparse.Namespace, out: Any) -> int:
    """Suggest a score floor for the catalogue."""
    from ..calibrate import calibrate_floor

    broker = _load_broker(args)
    broker.index()
    report = calibrate_floor(
        broker,
        noise=_read_lines(args.noise),
        samples=_read_lines(args.samples) or (),
        k=args.k,
    )

    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2), file=out)
        return EXIT_OK
    print(report.summary(), file=out)
    return EXIT_OK


def _cmd_doctor(args: argparse.Namespace, out: Any) -> int:
    """Report tools the catalogue cannot surface."""
    from ..diagnose import diagnose_catalogue

    broker = _load_broker(args)
    broker.index()
    report = diagnose_catalogue(
        broker,
        k=args.k,
        samples=_read_lines(args.samples) or (),
        margin=args.margin,
        pairs=args.pairs,
    )

    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2), file=out)
        return EXIT_OK
    print(report.summary(), file=out)
    return EXIT_OK


def _cmd_diff(args: argparse.Namespace, out: Any) -> int:
    """Show how a policy change moves each agent's reach."""
    from ..config import ToolBrokerConfig
    from ..reachability import diff_policies

    # The catalogue comes from the *proposed* config: a policy diff is only
    # meaningful against the tools it will actually be applied to.
    after_config = ToolBrokerConfig.from_file(args.after)
    broker = after_config.build()
    broker.index()

    diff = diff_policies(
        ToolBrokerConfig.from_file(args.before).policy.build(),
        after_config.policy.build(),
        broker.tools(),
    )

    if args.json:
        print(json.dumps(diff.model_dump(mode="json"), indent=2), file=out)
    else:
        print(diff.summary(), file=out)

    if args.fail_on_high_risk and diff.grants_high_risk:
        return EXIT_ERROR
    if args.fail_on_grant and any(delta.gained for delta in diff.deltas):
        return EXIT_ERROR
    return EXIT_OK


def _cmd_serve(args: argparse.Namespace, out: Any) -> int:
    """Run the MCP proxy."""
    from ..proxy.server import serve

    broker = _load_broker(args)
    broker.index()
    refreshing = f", refreshing every {args.refresh}s" if args.refresh else ""
    print(
        f"serving {len(broker)} tools over MCP ({args.transport}{refreshing})",
        file=sys.stderr,
    )
    serve(
        broker,
        transport=args.transport,
        host=args.host,
        port=args.port,
        default_k=args.k,
        refresh_interval=args.refresh,
    )
    return EXIT_OK


_COMMANDS = {
    "index": _cmd_index,
    "query": _cmd_query,
    "bench": _cmd_bench,
    "calibrate": _cmd_calibrate,
    "doctor": _cmd_doctor,
    "diff": _cmd_diff,
    "serve": _cmd_serve,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "WARNING", json_output=args.json_logs)

    handler = _COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces this
        parser.error(f"unknown command {args.command!r}")

    try:
        return handler(args, sys.stdout)
    except ToolBrokerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
