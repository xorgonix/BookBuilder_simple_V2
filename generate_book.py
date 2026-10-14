#!/usr/bin/env python3
"""
generate_book.py — draft a complete book from a YAML table of contents.

WHAT THIS IS
------------
A standalone, unattended driver for the prose stage of the BookBuilder pipeline.
It takes a TOC YAML file, builds one prompt per chapter exactly the way
`builder.py` does, sends each prompt to an OpenAI-compatible chat endpoint
(LM Studio by default), and writes the results to disk chapter by chapter.

It exists because `builder.py` is interactive: it prints a banner, waits for
"Press Enter", and needs a terminal. This script does the same work headlessly so
a whole book can be generated in one command, watched, logged, and resumed.

PROMPT FIDELITY
---------------
The prompts are NOT re-implemented here. The script imports `builder.py` and
calls its real `build_dynamic_prompt()` and `format_yaml_value()`, so what the
model sees is byte-for-byte what the pipeline would produce. If `builder.py`
cannot be imported, the script says so and falls back to an equivalent built-in
builder rather than failing.

CONFIGURATION — THE .env FILE
-----------------------------
Endpoint settings live in a `.env` file sitting next to this script, never in the
source, because this repository is public. Copy `.env.example` to `.env` and edit
it. The file takes three settings, one per line:

  * LLM_BASE_URL — the OpenAI-compatible endpoint. Defaults to local LM Studio.
  * LLM_MODEL    — the model name exactly as the server reports it.
  * LLM_API_KEY  — an optional credential, sent as a bearer token when set.

`.env` is listed in .gitignore and must never be committed. Real environment
variables always take precedence over the file, so a shell export or a CI secret
overrides it. A missing `.env` is not an error — the built-in defaults are used.
A local LM Studio server typically needs no credential at all.

FEATURES
--------
  * Endpoint, model and credential read from `.env`, environment, or CLI flags.
  * Streaming output, so you can watch progress instead of guessing.
  * Resumable: chapters already on disk are skipped unless --overwrite is given.
  * Atomic writes: a chapter file only appears once it is complete.
  * Retries with exponential backoff on transient network/HTTP errors.
  * Detects a truncated response (finish_reason == "length") and retries with a
    larger budget, so long chapters are not silently cut off mid-sentence.
  * Detects "thinking" models that return an empty content field and explains
    the problem instead of writing an empty file.
  * Strips reasoning blocks, code fences, chatter, and duplicate title headings.
  * Recompiles final_book.md after every chapter.
  * Per-chapter word counts and timings, plus a summary table at the end.
  * Clean Ctrl-C handling that tells you how to resume.

USAGE
-----
    python generate_book.py                       # uses .env and the defaults below
    python generate_book.py --only 5              # just chapter 5
    python generate_book.py --overwrite           # redo everything
    python generate_book.py --model qwen/qwen3.5-9b
    python generate_book.py --dry-run             # show prompts, call nothing

NOTE ON THINKING MODELS
-----------------------
Some models (gemma-4-12b-it among them) emit a hidden reasoning/scratchpad
channel alongside the visible answer. They work fine — they just need a token
budget large enough for the scratchpad AND the prose. A thinking model that is
given too small a budget spends the whole allotment reasoning and emits no
visible text at all, which looks identical to a broken model or a silent
failure, but is purely a budgeting problem.

That is why this script reserves REASONING_ALLOWANCE extra tokens on top of the
prose budget for every chapter, and why the "empty response" path below tells you
to raise the budget rather than blaming the model. If you want to tune it, set
LLM_REASONING_ALLOWANCE in .env, or pass --max-tokens to override the total
outright. A non-thinking model simply leaves the allowance unused.

License: same as the rest of this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - dependency hint
    sys.exit("PyYAML is required:  pip install pyyaml")

try:
    from writing_styles import persona_block, resolve_style
except ImportError:  # pragma: no cover - registry is optional
    persona_block = None
    resolve_style = None


# ---------------------------------------------------------------------------
# Local configuration
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dotenv(path: str | None = None) -> str | None:
    """
    Populate os.environ from a local .env file, if one exists.

    Keeps the endpoint, model name and credential out of the source tree, which
    matters because this repository is public. Values already present in the
    environment always win, so a shell export or a CI secret is not overwritten
    by a stale file. A missing file is normal and not an error.

    Parsing is deliberately minimal — one setting per line, `#` for comments,
    optional surrounding quotes — which covers everything a local run needs
    without adding a dependency.
    """
    env_path = path or os.path.join(HERE, ".env")
    if not os.path.isfile(env_path):
        return None

    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            name, _, value = line.partition("=")
            name = name.strip()
            value = value.strip().strip('"').strip("'")

            if name and name not in os.environ:
                os.environ[name] = value

    return env_path


DOTENV_USED = load_dotenv()

# Defaults. The credential has no built-in value on purpose: it belongs in .env.
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gemma-4-12b-it")
DEFAULT_CREDENTIAL = os.environ.get("LLM_API_KEY", "")
DEFAULT_TOC = os.path.join(HERE, "example_toc.yaml")

# Token budgeting. English prose runs roughly 1.5-1.8 tokens per word. A thinking
# model also spends tokens on its scratchpad, so we add an allowance on top of
# the prose budget, otherwise the visible answer gets cut off mid-sentence.
TOKENS_PER_WORD = 1.9
# Extra tokens reserved on top of the prose budget, so a thinking model has room
# for its scratchpad before it starts writing. Override via .env if needed.
REASONING_ALLOWANCE = int(os.environ.get("LLM_REASONING_ALLOWANCE", "2500"))
MIN_MAX_TOKENS = 3000
LENGTH_RETRY_FACTOR = 1.6  # budget multiplier when a response is truncated

REQUEST_TIMEOUT = 3600  # seconds; a long chapter on a local GPU can take a while
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 5

# Reasoning tags some models emit inline. Stripped from the output if present.
REASONING_PATTERN = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE
)

# Conversational preamble some models add before the prose.
CHATTER_PREFIXES = (
    "here is",
    "here's",
    "certainly",
    "sure,",
    "of course",
    "chapter 1 —",
    "below is",
)


# ---------------------------------------------------------------------------
# Small console helpers
# ---------------------------------------------------------------------------


def banner(text: str) -> None:
    """Print a section header that stands out in a long log."""
    print()
    print("=" * 74)
    print(f"  {text}")
    print("=" * 74)


def log(message: str, *, level: str = "info") -> None:
    """Timestamped console line. Kept deliberately plain so logs stay diffable."""
    stamp = time.strftime("%H:%M:%S")
    marker = {"info": "  ", "ok": "ok", "warn": "!!", "err": "XX"}.get(level, "  ")
    print(f"[{stamp}] {marker} {message}", flush=True)


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """Everything the run needs, resolved from CLI flags over .env over defaults."""

    toc_path: str
    out_dir: str
    base_url: str
    model: str
    credential: str
    temperature: float
    max_tokens: int | None  # None => derive per chapter from `word_count`
    overwrite: bool
    only: int | None
    dry_run: bool
    verbose: bool
    style: str | None = None  # literary voice override (e.g. didion, hemingway)


@dataclass
class Usage:
    """Token accounting for one request, as reported by the server."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Usage":
        """Build a Usage from an OpenAI-compatible `usage` object."""
        details = payload.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=int(payload.get("prompt_tokens") or 0),
            completion_tokens=int(payload.get("completion_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            total_tokens=int(payload.get("total_tokens") or 0),
        )

    @property
    def prose_tokens(self) -> int:
        """Output tokens that became visible prose rather than scratchpad."""
        return max(0, self.completion_tokens - self.reasoning_tokens)


@dataclass
class Completion:
    """One assistant reply, plus the transport facts we want to know about it."""

    text: str
    finish_reason: str = ""
    reasoning_chars: int = 0
    truncated: bool = False
    usage: Usage | None = None
    first_token_seconds: float | None = None


@dataclass
class ChapterResult:
    """What happened to one chapter, for the telemetry card and summary table."""

    number: int
    title: str
    words: int = 0
    seconds: float = 0.0
    status: str = "pending"  # written | skipped | failed | dry-run
    # Telemetry, filled in only for chapters this run actually generated.
    usage: Usage | None = None
    first_token_seconds: float | None = None
    tokens_per_second: float = 0.0


# ---------------------------------------------------------------------------
# TOC loading and validation
# ---------------------------------------------------------------------------


def load_toc(path: str) -> dict[str, Any]:
    """
    Read and validate a TOC file.

    Mirrors the contract enforced by builder.py: a YAML mapping with a
    top-level `chapters` list. Everything else is optional.
    """
    if not os.path.exists(path):
        sys.exit(f"TOC file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            sys.exit(f"TOC file is not valid YAML: {exc}")

    if not isinstance(data, dict):
        sys.exit("TOC file must be a YAML mapping at the top level.")
    if "chapters" not in data:
        sys.exit("TOC file is missing the required 'chapters' list.")
    if not isinstance(data["chapters"], list) or not data["chapters"]:
        sys.exit("'chapters' must be a non-empty list.")

    for index, chapter in enumerate(data["chapters"], start=1):
        if not isinstance(chapter, dict):
            sys.exit(f"Chapter {index} is not a mapping.")

    return data


def collapse(text: Any) -> str:
    """Squash a YAML block scalar into a single clean line for prompt context."""
    if text is None:
        return ""
    return " ".join(str(text).split())


# ---------------------------------------------------------------------------
# LM Studio / OpenAI-compatible client
# ---------------------------------------------------------------------------


def build_chat_payload(
    model: str, prompt: str, temperature: float, max_tokens: int, stream: bool
) -> dict[str, Any]:
    """Assemble a chat-completions request body."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        # Ask for a final usage block so we can report real token counts.
        "stream_options": {"include_usage": True},
    }


def request_headers(settings: Settings) -> dict[str, str]:
    """
    Standard headers for the endpoint.

    The Authorization header is omitted entirely when no credential is
    configured, rather than sending an empty bearer token.
    """
    headers = {"Content-Type": "application/json"}
    if settings.credential:
        headers["Authorization"] = f"Bearer {settings.credential}"
    return headers


def _stream_once(
    settings: Settings,
    prompt: str,
    max_tokens: int,
    on_progress: Callable[[int], None] | None,
) -> Completion:
    """
    Perform a single streaming request and collect the reply.

    Only the visible `content` channel is accumulated. A reasoning/scratchpad
    channel is measured but deliberately ignored, because it is not prose.
    """
    url = settings.base_url.rstrip("/") + "/chat/completions"
    payload = build_chat_payload(
        settings.model, prompt, settings.temperature, max_tokens, stream=True
    )

    pieces: list[str] = []
    reasoning_chars = 0
    finish_reason = ""
    visible_chars = 0
    first_token_seconds: float | None = None
    usage: Usage | None = None
    request_start = time.time()

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers(settings),
    )

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue  # SSE comment, keep-alive, or blank separator

            body = line[len("data:") :].strip()
            if body == "[DONE]":
                break

            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue  # partial frame

            # The server sends one final frame carrying the usage block.
            if event.get("usage"):
                usage = Usage.from_payload(event["usage"])

            choices = event.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

            delta = choice.get("delta") or {}

            # The scratchpad channel, when the server sends one.
            for key in ("reasoning_content", "reasoning"):
                scratch = delta.get(key)
                if scratch:
                    reasoning_chars += len(scratch)

            piece = delta.get("content")
            if piece:
                if first_token_seconds is None:
                    first_token_seconds = time.time() - request_start
                pieces.append(piece)
                visible_chars += len(piece)
                if on_progress:
                    on_progress(visible_chars)

    text = "".join(pieces).strip()
    return Completion(
        text=text,
        finish_reason=finish_reason,
        reasoning_chars=reasoning_chars,
        truncated=(finish_reason == "length"),
        usage=usage,
        first_token_seconds=first_token_seconds,
    )


def call_chat_completion(
    settings: Settings,
    prompt: str,
    max_tokens: int,
    *,
    on_progress: Callable[[int], None] | None = None,
) -> Completion:
    """
    Send one prompt and return a Completion.

    Retries transient failures with exponential backoff. Distinguishes an empty
    visible answer from a genuine transport failure, because the two have
    completely different fixes.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = _stream_once(settings, prompt, max_tokens, on_progress)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                wait = BACKOFF_SECONDS * attempt
                log(
                    f"attempt {attempt}/{MAX_ATTEMPTS} failed ({exc}); retrying in {wait}s",
                    level="warn",
                )
                time.sleep(wait)
                continue
            break

        if result.text:
            return result

        # Empty visible content is worth distinguishing from a plain failure: it
        # almost always means a thinking model consumed the whole budget on its
        # scratchpad, which is a settings problem rather than a model fault.
        if result.reasoning_chars:
            raise RuntimeError(
                "the model spent its entire token budget on the reasoning channel and "
                "produced no visible text. This is a budget problem, not a model fault. "
                "Raise the budget with --max-tokens, or raise LLM_REASONING_ALLOWANCE "
                "in .env, so tokens remain for the prose. "
                f"Scratchpad characters received: {result.reasoning_chars}."
            )

        last_error = RuntimeError("endpoint returned an empty completion")
        if attempt < MAX_ATTEMPTS:
            wait = BACKOFF_SECONDS * attempt
            log(f"empty response; retrying in {wait}s", level="warn")
            time.sleep(wait)

    raise RuntimeError(
        f"no usable completion after {MAX_ATTEMPTS} attempts. "
        f"Endpoint: {settings.base_url}  Model: {settings.model}.  Last error: {last_error}"
    )


def generate_chapter_text(
    settings: Settings,
    prompt: str,
    budget: int,
    *,
    on_progress: Callable[[int], None] | None = None,
) -> Completion:
    """
    Get one chapter's prose, growing the budget if the model was cut off.

    A thinking model that runs out of tokens mid-chapter would otherwise leave a
    half-finished file on disk, which looks like a content problem rather than a
    settings problem. One retry at a larger budget removes that failure mode.
    """
    result = call_chat_completion(settings, prompt, budget, on_progress=on_progress)

    if result.truncated and result.text:
        bigger = int(budget * LENGTH_RETRY_FACTOR)
        log(f"retrying chapter with a larger budget ({budget} -> {bigger} tokens)", level="warn")
        retry = call_chat_completion(settings, prompt, bigger, on_progress=on_progress)
        if retry.text and len(retry.text) > len(result.text):
            return retry

    return result


# ---------------------------------------------------------------------------
# Prompt building — reuse builder.py so prompts match the pipeline exactly
# ---------------------------------------------------------------------------


def load_prompt_builder() -> tuple[Callable[..., str], str]:
    """
    Return (build_dynamic_prompt, origin_description).

    Prefers the real implementation from builder.py. Falls back to an
    equivalent local copy so the script still runs if builder.py is absent.
    """
    try:
        sys.path.insert(0, HERE)
        import builder  # type: ignore  # the repo's own module

        if hasattr(builder, "build_dynamic_prompt"):
            return builder.build_dynamic_prompt, "builder.py (real pipeline code)"
    except Exception as exc:  # pragma: no cover - defensive
        log(f"could not import builder.py ({exc}); using built-in fallback", level="warn")

    return _fallback_build_prompt, "built-in fallback (builder.py unavailable)"


def _format_value(value: Any) -> str:
    """Flatten a YAML value to prompt text, matching builder.py's behaviour."""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "None"
    if isinstance(value, dict):
        return "; ".join(
            f"{k.replace('_', ' ').title()}: {_format_value(v)}" for k, v in value.items()
        )
    if value is None or value == "":
        return "Not specified"
    return str(value)


def _fallback_build_prompt(
    chapter: dict[str, Any],
    root: dict[str, Any],
    prev_summary: str,
    next_summary: str,
    style_override: str | None = None,
) -> str:
    """Equivalent to builder.build_dynamic_prompt, used only as a safety net."""
    metadata = root.get("metadata_defaults", {}) or {}
    title = root.get("title") or metadata.get("title", "Untitled Book")
    tone = root.get("tone") or metadata.get("tone", "Not specified")

    if resolve_style is not None:
        preset, raw_style = resolve_style(chapter, root, style_override)
    else:
        preset, raw_style = None, style_override

    chapter_num = chapter.get("chapter_number", "?")
    chapter_title = chapter.get("title", "Untitled")

    parts = [f'You are an expert novelist writing a book titled "{title}".', ""]

    if preset is not None and persona_block is not None:
        parts.extend(persona_block(preset))
    else:
        parts.append(f"Voice: {raw_style or 'Not specified'}")
        parts.append(f"Tone: {tone}")
        parts.append("")

    parts.extend(
        [
            "--- CURRENT ASSIGNMENT ---",
            f"Writing: CHAPTER {chapter_num}: {chapter_title}",
            f"Tone Target: {tone}",
            "",
            "--- FLOW CONTEXT ---",
            f"Previous Chapter Summary: {prev_summary}",
            f"Next Chapter Summary: {next_summary}",
            "",
            "--- CHAPTER SPECIFICATIONS (Dynamically Loaded) ---",
        ]
    )

    for key, value in chapter.items():
        if key in ("chapter_number", "title", "style", "voice"):
            continue
        parts.append(f"{key.replace('_', ' ').title()}: {_format_value(value)}")

    instructions = root.get("ai_instructions") or {}
    if instructions:
        parts.append("")
        parts.append("--- AUTHOR GUIDELINES (From YAML) ---")
        for section, rules in instructions.items():
            parts.append(f"{section.replace('_', ' ').title()}:")
            if isinstance(rules, list):
                parts.extend(f"- {rule}" for rule in rules)
            elif isinstance(rules, dict):
                parts.extend(f"- {k.replace('_', ' ').title()}: {v}" for k, v in rules.items())

    parts.extend(
        [
            "",
            "--- INSTRUCTIONS ---",
            f"Write the full prose text for Chapter {chapter_num} based strictly on the specifications above.",
            "Ensure a seamless transition from the previous chapter and set up the next.",
            "Maintain the exact voice cadence, syntax patterns, and restraint specified in the voice persona.",
            "Output ONLY the chapter prose in Markdown format. No title headings, introductions, or conversational filler.",
        ]
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------


def normalise_heading(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace, for heading comparison."""
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def clean_completion(text: str, self_title: str = "") -> str:
    """
    Remove everything the model wraps around the prose, keeping only the prose.

    Strips reasoning tags, an outer code fence, conversational preamble, the
    model's own "Chapter N" line, and a duplicate title heading. The saved file
    supplies its own "## Chapter N: Title" heading, so leaving the model's
    version in would put the chapter title into the book's contents twice.
    """
    text = REASONING_PATTERN.sub("", text)

    # Unwrap the whole response if the model fenced it.
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        text = "\n".join(lines)

    lines = text.strip().splitlines()

    # Drop leading blank lines, filler, and any "Chapter N" heading.
    while lines:
        head = lines[0].strip()
        lowered = head.lower()
        if not head:
            lines.pop(0)
        elif any(lowered.startswith(prefix) for prefix in CHATTER_PREFIXES):
            lines.pop(0)
        elif re.match(r"^#{0,6}\s*chapter\s+\d+\b", lowered):
            lines.pop(0)
        else:
            break

    # Drop a heading that merely repeats the chapter title, which is the very
    # next thing models emit after the "Chapter N" line.
    wanted = normalise_heading(self_title) if self_title else ""
    while lines:
        head = lines[0].strip()
        if not head:
            lines.pop(0)
            continue
        if re.match(r"^#{1,6}\s+\S", head) and wanted:
            candidate = normalise_heading(re.sub(r"^#{1,6}\s*", "", head))
            if candidate == wanted:
                lines.pop(0)
                continue
        break

    # Tidy any trailing horizontal rule the model signed off with.
    while lines and re.fullmatch(r"[-*_]{3,}", lines[-1].strip()):
        lines.pop()

    return "\n".join(lines).strip()


def slugify(title: str) -> str:
    """Match builder.py's output-folder naming so paths agree between scripts."""
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)
    return safe.replace(" ", "_").lower() or "output"


# ---------------------------------------------------------------------------
# Chapter rendering
# ---------------------------------------------------------------------------


def render_chapter(
    chapter: dict[str, Any],
    root: dict[str, Any],
    index: int,
    builder_fn: Callable[..., str],
    style: str | None = None,
) -> tuple[int, str, str]:
    """Return (chapter_number, title, prompt) for one chapter entry."""
    chapters = root["chapters"]
    number = chapter.get("chapter_number", index + 1)
    title = chapter.get("title", "Untitled")

    prev_raw = chapters[index - 1].get("summary") if index > 0 else "None (Start of book)"
    next_raw = (
        chapters[index + 1].get("summary")
        if index < len(chapters) - 1
        else "None (End of book)"
    )

    try:
        # Preferred: pass the style override through to the builder.
        prompt = builder_fn(
            chapter, root, collapse(prev_raw), collapse(next_raw), style_override=style
        )
    except TypeError:
        # Fallback for an external builder.py that lacks style_override support.
        prompt = builder_fn(chapter, root, collapse(prev_raw), collapse(next_raw))

    return int(number), str(title), prompt


def token_budget(chapter: dict[str, Any], override: int | None) -> int:
    """Token ceiling per chapter: explicit override, else derived from its word target."""
    if override:
        return override
    target_words = chapter.get("word_count") or 1500
    prose = int(target_words * TOKENS_PER_WORD)
    return max(MIN_MAX_TOKENS, prose + REASONING_ALLOWANCE)


def write_chapter(out_dir: str, number: int, title: str, prose: str) -> str:
    """
    Write one chapter file.

    The write is atomic: the target path only appears once the text is complete,
    so an interrupted run never leaves a half-written chapter behind.
    """
    chapters_dir = os.path.join(out_dir, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)
    target = os.path.join(chapters_dir, f"chapter_{number:03d}.md")

    body = f"## Chapter {number}: {title}\n\n{prose}\n"
    temporary = target + ".part"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(body)
    os.replace(temporary, target)

    return target


def recompile_book(out_dir: str, title: str) -> str | None:
    """Concatenate every chapter on disk into final_book.md, in order."""
    chapters_dir = os.path.join(out_dir, "chapters")
    if not os.path.isdir(chapters_dir):
        return None

    files = sorted(name for name in os.listdir(chapters_dir) if name.endswith(".md"))
    if not files:
        return None

    output = os.path.join(out_dir, "final_book.md")
    with open(output, "w", encoding="utf-8") as out_handle:
        out_handle.write(f"# {title}\n\n")
        for name in files:
            with open(os.path.join(chapters_dir, name), "r", encoding="utf-8") as in_handle:
                out_handle.write(in_handle.read().rstrip() + "\n\n")

    return output


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_chapter_metrics(
    number: int,
    title: str,
    words: int,
    seconds: float,
    completion: Completion,
    *,
    truncated: bool = False,
) -> None:
    """
    Print a per-chapter telemetry card: tokens and timings, labelled clearly.

    Everything shown comes from the server's own usage block plus wall-clock
    timings, so the numbers are measured rather than estimated. Missing values
    are reported as "n/a" instead of being guessed at.
    """
    usage = completion.usage
    width = 62

    def rule(char: str = "-") -> str:
        return "  " + char * width

    def row(label: str, value: str, note: str = "") -> str:
        tail = f"   {note}" if note else ""
        return f"  {label:<26} {value:<12}{tail}"

    # Header with the chapter fingerprint.
    print()
    print(rule("=" if not truncated else "!"))
    print(f"  CHAPTER {number}  ·  {title}")
    print(rule("=" if not truncated else "!"))

    # --- output ------------------------------------------------------------
    print("  OUTPUT")
    print(row("Prose written", f"{words:,} words"))
    if usage:
        print(row("  output tokens", f"{usage.prose_tokens:,}"))
        chars = len(completion.text)
        if usage.prose_tokens:
            print(row("  chars per token", f"{chars / usage.prose_tokens:.1f}"))
    if completion.reasoning_chars:
        print(row("Reasoning chars", f"{completion.reasoning_chars:,}"))

    # --- tokens ------------------------------------------------------------
    print(rule())
    print("  TOKENS")
    if usage:
        print(row("Prompt tokens", f"{usage.prompt_tokens:,}", "input"))
        print(row("Completion tokens", f"{usage.completion_tokens:,}", "all output"))
        print(row("  reasoning portion", f"{usage.reasoning_tokens:,}", "scratchpad"))
        print(row("  visible prose portion", f"{usage.prose_tokens:,}", "the writing"))
        print(row("Total tokens", f"{usage.total_tokens:,}"))
    else:
        print(row("Token counts", "n/a", "server sent no usage block"))

    # --- time --------------------------------------------------------------
    print(rule())
    print("  TIME")
    print(row("Total wall clock", f"{seconds:.1f}s"))
    if completion.first_token_seconds is not None:
        print(row("Time to first token", f"{completion.first_token_seconds:.2f}s"))
        generation_seconds = max(0.0, seconds - completion.first_token_seconds)
        print(row("Generation time", f"{generation_seconds:.1f}s", "first token to end"))
        if usage and generation_seconds > 0:
            print(
                row(
                    "Throughput (all out)",
                    f"{usage.completion_tokens / generation_seconds:.1f} tok/s",
                    "incl. scratchpad",
                )
            )
            print(
                row(
                    "Throughput (prose)",
                    f"{usage.prose_tokens / generation_seconds:.1f} tok/s",
                    "visible text only",
                )
            )
    if seconds > 0 and words:
        print(row("Prose rate", f"{words / seconds * 60:.0f} words/min", "wall clock"))

    # --- cost --------------------------------------------------------------
    if usage:
        print(rule())
        print("  COST")
        print(
            row(
                "Local inference",
                "free",
                "your own hardware" if not settings_are_remote() else "self-hosted endpoint",
            )
        )

    if truncated:
        print(rule("!"))
        print("  !!  OUTPUT MAY BE TRUNCATED — the token ceiling was reached.")
    print(rule("=" if not truncated else "!"))
    print()


def settings_are_remote() -> bool:
    """True if the configured endpoint is not on this machine."""
    host = (CURRENT_BASE_URL or "").lower()
    return not any(marker in host for marker in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))


# Set by run() so the reporting helpers can describe the endpoint honestly.
CURRENT_BASE_URL = DEFAULT_BASE_URL


def report_run_summary(results: list[ChapterResult], elapsed: float) -> None:
    """Print the closing table: per-chapter words, tokens and times, plus totals."""
    written = [r for r in results if r.status == "written"]
    failures = [r for r in results if r.status == "failed"]

    banner("Run summary")
    header = (
        f"  {'CH':>3}  {'WORDS':>6}  {'OUT TOK':>8}  {'REASON':>7}  "
        f"{'TIME':>7}  {'TOK/S':>6}  {'STATUS':<8}  TITLE"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        usage = r.usage
        out_tok = f"{usage.completion_tokens:,}" if usage else "-"
        reason = f"{usage.reasoning_tokens:,}" if usage else "-"
        secs = f"{r.seconds:.0f}s" if r.seconds else "-"
        tps = f"{r.tokens_per_second:.1f}" if r.tokens_per_second else "-"
        words = f"{r.words:,}" if r.words else "-"
        print(
            f"  {r.number:>3}  {words:>6}  {out_tok:>8}  {reason:>7}  "
            f"{secs:>7}  {tps:>6}  {r.status:<8}  {r.title}"
        )

    # Totals across the chapters this run actually generated.
    total_words = sum(r.words for r in written)
    total_seconds = sum(r.seconds for r in written)
    totals = [r.usage for r in written if r.usage]

    print()
    print("  " + "=" * (len(header) - 2))
    print("  TOTALS")
    print(f"  {'Chapters generated':<26} {len(written)}")
    print(f"  {'Failed':<26} {len(failures)}")
    print(f"  {'Words':<26} {total_words:,}")
    if totals:
        print(f"  {'Prompt tokens':<26} {sum(u.prompt_tokens for u in totals):,}")
        print(f"  {'Completion tokens':<26} {sum(u.completion_tokens for u in totals):,}")
        print(f"  {'  reasoning portion':<26} {sum(u.reasoning_tokens for u in totals):,}")
        print(f"  {'  visible prose portion':<26} {sum(u.prose_tokens for u in totals):,}")
        print(f"  {'Total tokens':<26} {sum(u.total_tokens for u in totals):,}")
    print(f"  {'Generation time':<26} {total_seconds / 60:.1f} min")
    print(f"  {'Overall elapsed':<26} {elapsed / 60:.1f} min")
    if total_seconds > 0 and total_words:
        print(
            f"  {'Overall prose rate':<26} "
            f"{total_words / total_seconds * 60:.0f} words/min"
        )
    print("  " + "=" * (len(header) - 2))
    print()

    if written:
        log(f"book file: {os.path.join(CURRENT_OUT_DIR, 'final_book.md')}", level="ok")
    if failures:
        log("re-run the same command to retry only the missing chapters", level="warn")


# Set by run() so the reporting helpers can name the output path.
CURRENT_OUT_DIR = ""


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(settings: Settings) -> int:
    """Generate the book. Returns a process exit code."""
    global CURRENT_BASE_URL, CURRENT_OUT_DIR
    CURRENT_BASE_URL = settings.base_url

    banner("BookBuilder — headless chapter generator")

    root = load_toc(settings.toc_path)
    chapters: list[dict[str, Any]] = root["chapters"]
    title = (
        root.get("title")
        or (root.get("metadata_defaults") or {}).get("title")
        or "Untitled Book"
    )

    builder_fn, builder_origin = load_prompt_builder()

    out_dir = os.path.abspath(settings.out_dir or os.path.join(HERE, slugify(title)))
    CURRENT_OUT_DIR = out_dir
    chapters_dir = os.path.join(out_dir, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)

    log(f"TOC           : {settings.toc_path}")
    log(f"Title         : {title}")
    log(f"Chapters      : {len(chapters)}")
    log(f"Endpoint      : {settings.base_url}")
    log(f"Model         : {settings.model}")
    log(f"Config file   : {DOTENV_USED or 'none (using defaults/environment)'}")
    log(f"Prompt source : {builder_origin}")
    log(f"Output        : {out_dir}")
    if settings.dry_run:
        log("DRY RUN — no requests will be sent", level="warn")

    results: list[ChapterResult] = []
    started = time.time()

    for index, chapter in enumerate(chapters):
        number, chapter_title, prompt = render_chapter(
            chapter, root, index, builder_fn, style=settings.style
        )
        target = os.path.join(chapters_dir, f"chapter_{number:03d}.md")
        result = ChapterResult(number=number, title=chapter_title)

        # --only: skip everything that isn't the requested chapter.
        if settings.only is not None and number != settings.only:
            result.status = "skipped"
            results.append(result)
            continue

        # Resume: a non-trivial existing file means this chapter is done.
        if not settings.overwrite and os.path.exists(target) and os.path.getsize(target) > 500:
            with open(target, encoding="utf-8") as handle:
                words = len(handle.read().split())
            result.words, result.status = words, "skipped"
            log(f"chapter {number:>2} '{chapter_title}' already drafted ({words} words) — skipping")
            results.append(result)
            continue

        if settings.dry_run:
            result.status = "dry-run"
            log(f"chapter {number:>2} '{chapter_title}' — prompt is {len(prompt)} chars")
            if settings.verbose:
                print("-" * 74)
                print(prompt)
                print("-" * 74)
            results.append(result)
            continue

        budget = token_budget(chapter, settings.max_tokens)
        log(f"chapter {number:>2} '{chapter_title}' — writing (budget {budget} tokens)…")
        chapter_start = time.time()

        def progress(chars: int, _n: int = number) -> None:
            # Print a heartbeat roughly every 800 characters of visible prose.
            if chars % 800 < 40:
                print(f"        … chapter {_n}: {chars} chars", flush=True)

        try:
            completion = generate_chapter_text(
                settings, prompt, budget, on_progress=progress if settings.verbose else None
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            result.status = "failed"
            log(f"chapter {number} failed: {exc}", level="err")
            results.append(result)
            continue

        prose = clean_completion(completion.text, chapter_title)
        if not prose:
            result.status = "failed"
            log(f"chapter {number} produced no usable text", level="err")
            results.append(result)
            continue

        write_chapter(out_dir, number, chapter_title, prose)
        recompile_book(out_dir, title)

        result.words = len(prose.split())
        result.seconds = time.time() - chapter_start
        result.status = "written"
        result.usage = completion.usage
        result.first_token_seconds = completion.first_token_seconds

        generation_seconds = (
            result.seconds - completion.first_token_seconds
            if completion.first_token_seconds is not None
            else result.seconds
        )
        if completion.usage and generation_seconds > 0:
            result.tokens_per_second = completion.usage.completion_tokens / generation_seconds

        log(f"chapter {number:>2} '{chapter_title}' written", level="ok")
        report_chapter_metrics(
            number,
            chapter_title,
            result.words,
            result.seconds,
            completion,
            truncated=completion.truncated,
        )
        results.append(result)

    # ---------------------------------------------------------------- summary
    elapsed = time.time() - started
    report_run_summary(results, elapsed)

    failures = [r for r in results if r.status == "failed"]
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> Settings:
    """Turn command-line arguments into a resolved Settings object."""
    parser = argparse.ArgumentParser(
        description="Draft a book from a YAML table of contents using an OpenAI-compatible endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "configuration:\n"
            "  Endpoint, model and credential are read from .env (see .env.example),\n"
            "  then overridden by the flags below.\n"
            "\n"
            "examples:\n"
            "  python generate_book.py\n"
            "  python generate_book.py --model qwen/qwen3.5-9b --verbose\n"
            "  python generate_book.py --only 3 --overwrite\n"
            "  python generate_book.py --dry-run\n"
        ),
    )
    parser.add_argument("--toc", default=DEFAULT_TOC, help="path to the TOC YAML file")
    parser.add_argument("--out", default=None, help="output directory (default: a folder named after the book)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name as the server reports it")
    parser.add_argument(
        "--api-key",
        dest="credential",
        default=DEFAULT_CREDENTIAL,
        help="credential for the endpoint (default: read from the environment)",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="hard token ceiling per chapter")
    parser.add_argument("--only", type=int, default=None, help="generate only this chapter number")
    parser.add_argument("--overwrite", action="store_true", help="regenerate chapters that already exist")
    parser.add_argument("--dry-run", action="store_true", help="build prompts but send nothing")
    parser.add_argument("--verbose", action="store_true", help="stream progress and print prompts in dry runs")
    parser.add_argument(
        "--style",
        default=None,
        help="literary voice override (e.g. didion, hemingway, mccarthy, austen, vonnegut)",
    )
    args = parser.parse_args(argv)

    return Settings(
        toc_path=os.path.abspath(args.toc),
        out_dir=os.path.abspath(args.out) if args.out else "",
        base_url=args.base_url,
        model=args.model,
        credential=args.credential,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite,
        only=args.only,
        dry_run=args.dry_run,
        verbose=args.verbose,
        style=args.style,
    )


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point. Handles Ctrl-C cleanly so a long run is never a mystery."""
    settings = parse_args(argv)
    try:
        return run(settings)
    except KeyboardInterrupt:
        print()
        log("interrupted — chapters already written are safe", level="warn")
        log("run the same command again to continue", level="warn")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
