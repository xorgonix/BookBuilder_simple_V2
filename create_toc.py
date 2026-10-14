"""
Generate a structured JSON book outline (TOC + metadata) via Ollama's
OpenAI-compatible API. Output is a single JSON document to stdout.

Usage (defaults are fine):
    python create_toc.py \
        --title "My Book" \
        --genre "Sci-Fi" \
        --voice "First person" \
        --tone "Witty" \
        --premise "A coder time-travels via version control." \
        --num-chapters 12 \
        --target-words 80000 \
        --format yaml \
        --out-dir output

NOTE: The --out-dir path is now always resolved relative to this script's
own directory (where this file lives) unless you provide an absolute path.
So running the script from any working directory produces output under:
    <directory_of_this_file>/<out-dir>/<slugified-title>/
The directory path is created recursively if missing.

Environment:
  OLLAMA_BASE_URL  (default: http://localhost:11434/v1)
  OLLAMA_API_KEY   (default: ollama)
  MODEL_NAME       (default: gemma-4-12b-it-aeon-32k:latest)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # Defer import error until actually requested

# ------------------------- Utilities -------------------------
def _slugify(name: str, max_len: int = 64) -> str:
    # Remove characters illegal on Windows paths and normalize spaces
    illegal = set('<>:"/\\|?*')
    cleaned = ''.join(ch for ch in name if ch not in illegal)
    cleaned = cleaned.strip().replace('\t', ' ')
    # Replace spaces with dashes, collapse repeats
    out = []
    prev_dash = False
    for ch in cleaned:
        if ch.isspace():
            if not prev_dash:
                out.append('-')
                prev_dash = True
        elif ch.isalnum() or ch in ('-', '_'):
            out.append(ch)
            prev_dash = (ch == '-')
        # else drop other punctuation
    slug = ''.join(out).strip('- .')
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip('-')
        print("slug: ", slug)
    return slug or 'book'

def _unique_run_dir(base_dir: str, title: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    base = _slugify(title)
    candidate = os.path.join(base_dir, base)
    if not os.path.exists(candidate):
        return candidate
    # Add numeric suffix to ensure a fresh run dir each time
    i = 1
    while True:
        cand = f"{candidate}-{i}"
        if not os.path.exists(cand):
            return cand
        i += 1

def _emit_files(plan_obj: 'BookPlan', out_dir: str) -> tuple[str, str | None]:
    data = plan_obj.model_dump(mode='python')
    run_dir = _unique_run_dir(out_dir, plan_obj.title)
    os.makedirs(run_dir, exist_ok=True)
    base_name = _slugify(plan_obj.title, max_len=80)
    json_path = os.path.join(run_dir, f"{base_name}-toc.json")
    yaml_path = os.path.join(run_dir, f"{base_name}-toc.yaml")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    wrote_yaml = None
    if yaml is not None:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        wrote_yaml = yaml_path
    return json_path, wrote_yaml


# --------------------------- Schema ---------------------------
class Chapter(BaseModel):
    number: int = Field(description="1-based chapter number")
    title: str
    length_words: int = Field(description="Estimated length in words for the chapter")
    purpose: str = Field(description="Narrative purpose of this chapter")
    summary: str = Field(description="3-7 sentence summary")
    location: str = Field(description="Primary setting/location")
    research_needed: List[str] = Field(description="List of research topics still needed")
    research_gathered: List[str] = Field(description="List of sources or facts already gathered")


class BookPlan(BaseModel):
    title: str
    genre: str
    voice: str
    tone: str
    premise: str
    num_chapters: int
    target_word_count: int
    chapters: List[Chapter]


# ------------------------- Prompting -------------------------
def build_messages(args: argparse.Namespace):
    sys_prompt = (
        "You are an expert editor and book development assistant. "
        "Return ONLY valid JSON conforming exactly to the provided schema. "
        "No Markdown, no commentary, no backticks. "
        "Ensure integers are numbers (not strings)."
    )

    user_prompt = (
        "Create a complete table of contents (TOC) and metadata for a book. "
        "Populate every required field in this schema:\n\n"
        "Top-level fields:\n"
        "- title (string)\n"
        "- genre (string)\n"
        "- voice (string)\n"
        "- tone (string)\n"
        "- premise (string)\n"
        "- num_chapters (integer)\n"
        "- target_word_count (integer)\n"
        "- chapters (array of Chapter)\n\n"
        "Chapter fields:\n"
        "- number (integer, 1-based)\n"
        "- title (string)\n"
        "- length_words (integer)\n"
        "- purpose (string)\n"
        "- summary (string, 3-7 sentences)\n"
        "- location (string)\n"
        "- research_needed (array of strings)\n"
        "- research_gathered (array of strings)\n\n"
        f"Use these guide parameters (may refine but must remain consistent):\n"
        f"- title: {args.title}\n"
        f"- genre: {args.genre}\n"
        f"- voice: {args.voice}\n"
        f"- tone: {args.tone}\n"
        f"- premise: {args.premise}\n"
        f"- num_chapters: {args.num_chapters}\n"
        f"- target_word_count: {args.target_words}\n\n"
        "Constraints:\n"
        "- chapters length_words should roughly sum to target_word_count.\n"
        "- chapter numbers must start at 1 and be sequential.\n"
        "- Output must be JSON matching the schema with no extra fields."
    )

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a JSON/YAML book TOC via Ollama OpenAI-compatible API")
    parser.add_argument("--title", default="Untitled Book")
    parser.add_argument("--genre", default="General Fiction")
    parser.add_argument("--voice", default="Third person limited")
    parser.add_argument("--tone", default="Engaging, clear, and modern")
    parser.add_argument("--premise", default="A protagonist undergoes a transformative journey.")
    parser.add_argument("--num-chapters", type=int, default=12)
    parser.add_argument("--target-words", type=int, default=80000)
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    parser.add_argument("--out-dir", default="output", help="Base output directory for run artifacts (resolved relative to script directory unless absolute). Can be overridden by env BOOK_OUTPUT_DIR when left as default.")
    args = parser.parse_args(argv)

    # Resolve output directory relative to the script's own directory so that
    # runs from different CWDs behave consistently.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_override = os.getenv("BOOK_OUTPUT_DIR")
    # Only apply env override if user didn't explicitly set a different out-dir
    if args.out_dir == "output" and env_override:
        # If env is relative, make it relative to script_dir for consistency
        resolved_out_dir = env_override if os.path.isabs(env_override) else os.path.join(script_dir, env_override)
    else:
        if os.path.isabs(args.out_dir):
            resolved_out_dir = args.out_dir
        else:
            resolved_out_dir = os.path.join(script_dir, args.out_dir)
    # Ensure base directory exists (sub run dirs created later as needed)
    os.makedirs(resolved_out_dir, exist_ok=True)

    # Stash back for downstream calls
    args.out_dir = resolved_out_dir
    print(f"[info] Base output directory: {args.out_dir}")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    model = os.getenv("MODEL_NAME", "gemma-4-12b-it-aeon-32k:latest")

    client = OpenAI(base_url=base_url, api_key=api_key)

    messages = build_messages(args)

    # Prefer strict Pydantic parsing when available.
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            temperature=0,
            messages=messages,  # type: ignore[arg-type]
            response_format=BookPlan,
        )
        msg = completion.choices[0].message
        if msg.parsed:
            # Always write files
            json_file, yaml_file = _emit_files(msg.parsed, args.out_dir)
            # Also stream to stdout in chosen format
            if args.format == "yaml":
                if yaml is None:
                    print(json.dumps({"error": "PyYAML not installed"}), file=sys.stderr)
                    return 3
                print(yaml.safe_dump(msg.parsed.model_dump(mode="python"), sort_keys=False))
            else:
                print(msg.parsed.model_dump_json(indent=2))
            print(f"\nWrote JSON: {json_file}")
            if yaml_file:
                print(f"Wrote YAML: {yaml_file}")
            return 0
        if msg.refusal:
            print(json.dumps({"error": "refused", "reason": msg.refusal}), file=sys.stderr)
            return 2
    except Exception as e:
        # Fall back to plain chat completion, still asking for raw JSON.
        pass

    try:
        alt = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,  # type: ignore[arg-type]
        )
        content = alt.choices[0].message.content or ""
        # Try loading as JSON, else print raw for caller-side handling.
        try:
            obj = json.loads(content)
            # Optionally validate against schema
            plan = BookPlan.model_validate(obj)
            # Always write files
            json_file, yaml_file = _emit_files(plan, args.out_dir)
            if args.format == "yaml":
                if yaml is None:
                    print(json.dumps({"error": "PyYAML not installed"}), file=sys.stderr)
                    return 3
                print(yaml.safe_dump(plan.model_dump(mode="python"), sort_keys=False))
            else:
                print(plan.model_dump_json(indent=2))
            print(f"\nWrote JSON: {json_file}")
            if yaml_file:
                print(f"Wrote YAML: {yaml_file}")
        except Exception:
            # Print raw content; caller can post-process.
            if args.format == "yaml":
                if yaml is None:
                    print(json.dumps({"error": "PyYAML not installed"}), file=sys.stderr)
                    return 3
                try:
                    print(yaml.safe_dump(json.loads(content), sort_keys=False))
                except Exception:
                    # Content wasn't JSON at all, emit as a YAML string
                    print(yaml.safe_dump(content))
            else:
                print(content)
        return 0
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
