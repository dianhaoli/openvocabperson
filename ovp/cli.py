"""Command-line interface (``python -m ovp``).

``serve`` runs the HTTP API + UI. ``analyze``, ``search`` and ``models`` use the same
application context as the server (database, file store, model registry, job manager) in
process, without HTTP, so results land in the same library the UI shows.

Results go to stdout (a table, or JSON with ``--json``); progress, logs and anything the
model libraries print go to stderr, so ``--json`` output can be piped safely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from . import __version__

if TYPE_CHECKING:  # heavy imports (torch, models) stay out of `--help` and `serve` start-up
    from .context import AppContext


class CliError(Exception):
    """A user-facing failure: printed without a traceback, exit code 2."""


# ──────────────────────────────────────────────────────────────────────────────
# plumbing
# ──────────────────────────────────────────────────────────────────────────────


@contextmanager
def stdout_reserved() -> Iterator[TextIO]:
    """Point file descriptor 1 at stderr for the duration and yield a stream to the real stdout.

    Model libraries (ultralytics, transformers, C extensions) print freely; redirecting at the
    descriptor level keeps every byte of that noise out of the command's result stream.
    """
    sys.stdout.flush()
    real_fd = os.dup(1)
    os.dup2(2, 1)
    out = os.fdopen(real_fd, "w", encoding="utf-8")
    try:
        yield out
    finally:
        sys.stdout.flush()
        out.flush()
        os.dup2(real_fd, 1)
        out.close()


@asynccontextmanager
async def app_context() -> AsyncIterator[AppContext]:
    """The server's AppContext, created and torn down for one command.

    Default models are loaded first, exactly as at server startup, so the same fallbacks apply
    (e.g. Re-ID OSNet → CLIP → colour histogram when weights cannot be downloaded).
    """
    from .context import create_context
    from .runtime import get_runtime

    try:
        ctx = await create_context()
    except OSError as e:
        raise CliError(f"cannot connect to PostgreSQL ({e}); check DATABASE_URL") from e
    try:
        await ctx.warmup()
        yield ctx
    finally:
        for job in ctx.jobs.jobs.values():
            ctx.jobs.cancel(job)
        await ctx.registry.aclose()
        await ctx.db.close()
        get_runtime().shutdown()


def write_json(out: TextIO, data: Any) -> None:
    json.dump(data, out, indent=2, default=str, ensure_ascii=False)
    out.write("\n")


def shorten(text: str | None, width: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def describe_person(entity: dict[str, Any]) -> str:
    person = entity.get("person")
    status = entity.get("match_status")
    if not person:
        return f"({status})" if status else ""
    name = person.get("label") or person["id"]
    score = entity.get("match_score")
    return f"{name} ({status}{f' {score:.2f}' if score is not None else ''})"


def print_entity(out: TextIO, e: dict[str, Any], indent: str = "  ") -> None:
    box = ",".join(f"{v:.0f}" for v in e["box"])
    when = f" t={e['t_start']:.1f}-{e['t_end']:.1f}s" if e.get("t_start") is not None else ""
    concept = f" [{e['concept']}]" if e.get("concept") and e["concept"] != e["label"] else ""
    verified = {True: " ✓", False: " ✗", None: ""}[e.get("verified")]
    out.write(f"{indent}{e['label']}{concept}{verified}  {e['score']:.2f}  box={box}{when}  {e['stage']}")
    person = describe_person(e)
    if person:
        out.write(f"  person={person}")
    out.write("\n")
    if e.get("description"):
        out.write(f"{indent}    {shorten(e['description'], 150)}\n")


# ──────────────────────────────────────────────────────────────────────────────
# serve
# ──────────────────────────────────────────────────────────────────────────────


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("ovp.api.app:app", host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# analyze
# ──────────────────────────────────────────────────────────────────────────────


def analyze_options(args: argparse.Namespace) -> Any:
    from .pipeline.analyzer import AnalyzeOptions

    return AnalyzeOptions.from_dict({
        "query": args.query,
        "expand": args.expand,
        "verify": args.verify,
        "segment": args.segment,
        "describe": args.describe,
        "detector": args.detector,
        "segmenter": args.segmenter,
        "vlm": args.vlm,
        "llm": args.llm,
        "conf": args.conf,
        "video_fps": args.video_fps,
        "max_frames": args.max_frames,
        "reid": not args.no_reid,
    })


class ProgressPrinter:
    """Turns job events into one-line progress messages on stderr."""

    def __init__(self, quiet: bool):
        self.quiet = quiet
        self.names: dict[str, str] = {}

    def __call__(self, ev: dict[str, Any]) -> None:
        kind = ev["type"]
        if kind == "media_created":
            self.names[ev["media"]["id"]] = ev["media"]["name"]
        if kind in ("log", "error") and ev.get("level", "error") in ("warning", "error"):
            self._say(f"{ev.get('level', 'error')}: {ev['message']}")
        if self.quiet:
            return
        if kind == "plan":
            plan = ev["plan"]
            prompts = "default COCO vocabulary" if plan["source"] == "default" else ", ".join(plan["prompts"])
            self._say(f"plan ({plan['source']}): {shorten(prompts, 200)}")
        elif kind == "media_started":
            self._say(f"[{ev['index'] + 1}/{ev['total']}] {ev['name']}")
        elif kind == "stage":
            self._say(f"    {self.names.get(ev['media_id'], ev['media_id'])}: {ev['message']}")
        elif kind == "media_done" and ev.get("media"):
            m = ev["media"]
            self._say(f"    {m['name']}: {len(m['entities'])} entities")

    @staticmethod
    def _say(text: str) -> None:
        print(text, file=sys.stderr, flush=True)


async def run_analyze(args: argparse.Namespace, out: TextIO) -> int:
    from . import serialize

    missing = [p for p in args.paths if not Path(p).expanduser().exists()]
    if missing:
        print(f"error: not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    opts = analyze_options(args)
    async with app_context() as ctx:
        items = ctx.jobs.collect_paths(args.paths)
        if not items:
            print("error: no images or videos found at the given paths", file=sys.stderr)
            return 2
        job = ctx.jobs.submit_analyze(items, opts)
        progress = ProgressPrinter(args.quiet)
        try:
            async for ev in ctx.jobs.subscribe(job):
                progress(ev)
            await ctx.jobs.wait(job)
        except asyncio.CancelledError:
            ctx.jobs.cancel(job)
            await ctx.jobs.wait(job)
            raise
        media = []
        for media_id in job.media_ids:
            row = await ctx.db.get_media(media_id)
            if row is not None:
                media.append(serialize.media_detail(row, await ctx.db.media_entities(media_id)))
    if args.json:
        write_json(out, {"job": job.to_dict(), "media": media})
    else:
        for m in media:
            persons = sum(1 for e in m["entities"] if e["label"] == "person")
            size = f"{m['width']}x{m['height']}" + (f", {m['duration']:.1f}s" if m.get("duration") else "")
            out.write(f"{m['name']}  {m['kind']} {size}  {len(m['entities'])} entities, {persons} person(s)  id={m['id']}\n")
            for e in sorted(m["entities"], key=lambda e: (e["label"] != "person", -e["score"])):
                print_entity(out, e)
        if job.error:
            out.write(f"job {job.status}: {job.error}\n")
    return 0 if job.status == "done" else 1


# ──────────────────────────────────────────────────────────────────────────────
# search
# ──────────────────────────────────────────────────────────────────────────────


async def run_search(args: argparse.Namespace, out: TextIO) -> int:
    image = None
    if args.image:
        from PIL import Image, ImageOps

        image = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    if not (args.text and args.text.strip()) and image is None:
        print("error: give search text and/or --image", file=sys.stderr)
        return 2
    filters = {"labels": args.label, "kind": args.kind, "media_id": args.media, "person_id": args.person}
    async with app_context() as ctx:
        hits = await ctx.search.search(args.text, image, filters, args.limit)
        results = await ctx.search.hydrate(hits)
    if args.json:
        write_json(out, {"results": results, "count": len(results), "query": {"text": args.text, "has_image": image is not None}})
        return 0
    if not results:
        out.write("no results\n")
    for rank, r in enumerate(results, 1):
        e, m = r["entity"], r["media"] or {}
        signals = " ".join(f"{k}={v:.2f}" for k, v in r["scores"].items())
        out.write(f"{rank:>3}. {r['score']:.4f}  {m.get('name', '?')}  ({signals})\n")
        print_entity(out, e, indent="       ")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# models
# ──────────────────────────────────────────────────────────────────────────────


async def run_models(args: argparse.Namespace, out: TextIO) -> int:
    from .models.registry import ROLES, get_registry
    from .runtime import get_runtime

    registry = get_registry()
    try:
        if args.check:
            for role in ROLES:
                await registry.ensure(role)
        report = registry.report()
    finally:
        await registry.aclose()
        get_runtime().shutdown()
    if args.json:
        write_json(out, report)
        return 0
    out.write(f"device: {report['device']}\n")
    caps = ", ".join(f"{k}={v}" for k, v in report["capabilities"].items())
    out.write(f"capabilities: {caps}\n")
    for role in ROLES:
        default = report["defaults"].get(role, "none")
        out.write(f"\n{role}  (default: {default})\n")
        for p in report["providers"].get(role, []):
            mark = "*" if p["spec"] == default else ("+" if p["available"] else "-")
            note = f"  – {p['note']}" if p.get("note") else ""
            out.write(f"  {mark} {p['spec']:<52} {p['label']}{note}\n")
    out.write("\n* default   + available   - unavailable\n")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# entry point
# ──────────────────────────────────────────────────────────────────────────────


def tri_state(value: str) -> bool | str:
    v = value.lower()
    if v not in ("auto", "true", "false"):
        raise argparse.ArgumentTypeError("expected auto, true or false")
    return v if v == "auto" else v == "true"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ovp", description="OpenVocab Vision: open-vocabulary detection, VLM description, Re-ID and search.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API and web UI")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(handler=cmd_serve)

    def output_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--json", action="store_true", help="print machine-readable JSON (same shapes as the HTTP API)")
        p.add_argument("-v", "--verbose", action="store_true", help="show INFO logs on stderr")

    analyze = sub.add_parser("analyze", help="analyze images/videos (files or folders) into the library")
    analyze.add_argument("paths", nargs="+", help="image/video files or folders (searched recursively)")
    analyze.add_argument("-q", "--query", help="what to find: free text, comma list, or abstract concept (default: COCO vocabulary)")
    analyze.add_argument("--describe", choices=("persons", "all", "none"), default="persons", help="which detections get a VLM description")
    analyze.add_argument("--segment", action="store_true", help="add segmentation masks")
    analyze.add_argument("--expand", type=tri_state, default="auto", metavar="{auto,true,false}", help="LLM-expand abstract queries into detector prompts")
    analyze.add_argument("--verify", type=tri_state, default="auto", metavar="{auto,true,false}", help="VLM yes/no check of candidates")
    analyze.add_argument("--detector", help="detector spec override, e.g. yoloe:yoloe-11s-seg.pt")
    analyze.add_argument("--segmenter", help="segmenter spec override, e.g. sam:mobile_sam.pt")
    analyze.add_argument("--vlm", help="VLM spec override, e.g. openai:gpt-4.1-mini")
    analyze.add_argument("--llm", help="LLM spec used for query planning")
    analyze.add_argument("--conf", type=float, help="detector confidence threshold")
    analyze.add_argument("--video-fps", type=float, help="frames per second sampled from videos")
    analyze.add_argument("--max-frames", type=int, help="maximum frames sampled per video")
    analyze.add_argument("--no-reid", action="store_true", help="do not link persons to identity clusters")
    analyze.add_argument("--quiet", action="store_true", help="no progress output (warnings and errors still shown)")
    output_flags(analyze)
    analyze.set_defaults(runner=run_analyze)

    search = sub.add_parser("search", help="hybrid search over the library")
    search.add_argument("text", nargs="?", help="search text, e.g. 'man in a red jacket'")
    search.add_argument("--image", help="query image (CLIP similarity + person Re-ID)")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--label", action="append", help="only these detector labels (repeatable)")
    search.add_argument("--kind", choices=("image", "video"))
    search.add_argument("--media", help="only entities of this media id")
    search.add_argument("--person", help="only sightings of this person id")
    output_flags(search)
    search.set_defaults(runner=run_search)

    models = sub.add_parser("models", help="list model providers, availability and defaults")
    models.add_argument("--check", action="store_true", help="load every default model to verify it works (downloads weights)")
    output_flags(models)
    models.set_defaults(runner=run_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return args.handler(args)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    with stdout_reserved() as out:
        try:
            return asyncio.run(args.runner(args, out))
        except KeyboardInterrupt:
            print("interrupted", file=sys.stderr)
            return 130
        except (CliError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
