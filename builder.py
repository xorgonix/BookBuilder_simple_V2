import argparse
import json
import os
import sys
import time

import requests
import yaml
from openai import OpenAI

try:
    from writing_styles import persona_block, resolve_style
except ImportError:  # pragma: no cover - registry is optional
    persona_block = None
    resolve_style = None

# ==========================================
# SETUP & CONFIGURATION
# ==========================================

# Any OpenAI-compatible endpoint works. Defaults to a local Ollama server.
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # local Ollama ignores this

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Override with the --model flag or the OLLAMA_MODEL environment variable.
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma-4-12b-it-aeon-32k:latest")
YAML_FILENAME = None
OUTPUT_DIR = None

book_state = {"yaml_data": None, "chapters_content": {}}


def get_output_dir():
    """Returns the output directory path based on the YAML title."""
    global OUTPUT_DIR
    if OUTPUT_DIR is None:
        title = book_state.get("yaml_data", {}).get("title")
        if title:
            safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)
            safe_title = safe_title.replace(" ", "_").lower()
            OUTPUT_DIR = safe_title
        else:
            OUTPUT_DIR = "output"
    return OUTPUT_DIR

# ==========================================
# HELPER FUNCTIONS
# ==========================================


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(text):
    print("\n" + "=" * 70)
    print(f" {text.upper()} ")
    print("=" * 70 + "\n")


def log_status(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return None


def make_llm_call(prompt, temperature=0.7, max_tokens=3000, context_label="LLM Call"):
    """Send one prompt. Returns (content, usage, duration_seconds)."""
    log_status(f"{context_label}: Sending request to Ollama...")
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        duration = round(time.time() - start_time, 2)
        content = response.choices[0].message.content.strip()
        usage = getattr(response, "usage", None)
        log_status(f"{context_label}: Response received.")
        return content, usage, duration
    except Exception as e:
        log_status(f"{context_label}: ERROR during LLM call: {e}")
        raise e


def format_usage_line(usage, duration):
    """One-line summary of the tokens spent and time taken for a single call."""
    if usage is None:
        return f"Tokens: n/a (server sent no usage block)  |  Time: {duration}s"

    total = getattr(usage, "total_tokens", None)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if total is None:
        return f"Tokens: n/a  |  Time: {duration}s"

    line = f"Tokens: {total:,} total ({prompt:,} in / {completion:,} out)"
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning = getattr(details, "reasoning_tokens", None)
        if reasoning:
            line += f", {reasoning:,} reasoning"
    return f"{line}  |  Time: {duration}s"


# ==========================================
# DYNAMIC PROMPT BUILDER
# ==========================================


def format_yaml_value(val, indent=0):
    """Recursively formats YAML values into readable prompt text."""
    prefix = "  " * indent
    if isinstance(val, list):
        return ", ".join(str(v) for v in val) if val else "None"
    elif isinstance(val, dict):
        lines = []
        for k, v in val.items():
            key_str = k.replace("_", " ").title()
            val_str = format_yaml_value(v, indent + 1)
            lines.append(f"{prefix}{key_str}: {val_str}")
        return "\n".join(lines)
    elif val is None or val == "":
        return "Not specified"
    else:
        return str(val)


def build_dynamic_prompt(chap_data, root_data, prev_sum, next_sum, style_override=None):
    """Constructs a prompt entirely from the YAML structure, injecting an
    optional literary-style persona when one is resolved."""
    title = root_data.get("title") or root_data.get("metadata_defaults", {}).get(
        "title", "Untitled Book"
    )
    tone = root_data.get("tone") or root_data.get("metadata_defaults", {}).get(
        "tone", "Not specified"
    )

    if resolve_style is not None:
        preset, raw_style = resolve_style(chap_data, root_data, style_override)
    else:  # registry unavailable: degrade to a freeform voice line
        preset, raw_style = None, style_override

    chapter_num = chap_data.get("chapter_number", "?")
    chapter_title = chap_data.get("title", "Untitled")

    lines = [f'You are an expert novelist writing a book titled "{title}".', ""]

    if preset is not None and persona_block is not None:
        lines.extend(persona_block(preset))
    else:
        lines.append(f"Voice: {raw_style or 'Not specified'}")
        lines.append(f"Tone: {tone}")
        lines.append("")

    lines.extend(
        [
            "--- CURRENT ASSIGNMENT ---",
            f"Writing: CHAPTER {chapter_num}: {chapter_title}",
            f"Tone Target: {tone}",
            "",
            "--- FLOW CONTEXT ---",
            f"Previous Chapter Summary: {prev_sum}",
            f"Next Chapter Summary: {next_sum}",
            "",
            "--- CHAPTER SPECIFICATIONS (Dynamically Loaded) ---",
        ]
    )

    # Inject every field from the chapter YAML dynamically (title/style/voice are
    # already represented above, so they are skipped here).
    for key, value in chap_data.items():
        if key in ("chapter_number", "title", "style", "voice"):
            continue
        lines.append(f"{key.replace('_', ' ').title()}: {format_yaml_value(value)}")

    # Inject root-level AI instructions if they exist
    ai_instructions = root_data.get("ai_instructions", {})
    if ai_instructions:
        lines.append("")
        lines.append("--- AUTHOR GUIDELINES (From YAML) ---")
        for section, rules in ai_instructions.items():
            lines.append(f"{section.replace('_', ' ').title()}:")
            if isinstance(rules, list):
                lines.extend(f"- {rule}" for rule in rules)
            elif isinstance(rules, dict):
                lines.extend(
                    f"- {k.replace('_', ' ').title()}: {v}" for k, v in rules.items()
                )

    lines.extend(
        [
            "",
            "--- INSTRUCTIONS ---",
            f"Write the full prose text for Chapter {chapter_num} based strictly on the specifications above.",
            "Ensure a seamless transition from the previous chapter and set up the next.",
            "Maintain the exact voice cadence, syntax patterns, and restraint specified in the voice persona.",
            "Output ONLY the chapter prose in Markdown format. No title headings, introductions, or conversational filler.",
        ]
    )

    return "\n".join(lines)


# ==========================================
# INCREMENTAL SAVE SYSTEM
# ==========================================


def save_chapter_to_disk(chap_num, content, chap_title=""):
    """Saves individual chapter and recompiles the main book file immediately."""
    output_dir = get_output_dir()
    chapters_dir = os.path.join(output_dir, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)
    filename = f"chapter_{chap_num:03d}.md"
    filepath = os.path.join(chapters_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"## Chapter {chap_num}: {chap_title}\n\n{content}\n")
    log_status(f"Saved Chapter {chap_num} to {filepath}")

    compile_final_book()


def compile_final_book():
    """Reads all saved chapters in order and updates the main markdown file."""
    output_dir = get_output_dir()
    chapters_dir = os.path.join(output_dir, "chapters")
    output_file = os.path.join(output_dir, "final_book.md")

    if not os.path.exists(chapters_dir):
        return

    title = book_state["yaml_data"].get("title") or book_state["yaml_data"].get(
        "metadata_defaults", {}
    ).get("title", "Untitled Book")
    chap_files = sorted([f for f in os.listdir(chapters_dir) if f.endswith(".md")])

    if not chap_files:
        return

    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write(f"# {title}\n\n")
        for cf in chap_files:
            filepath = os.path.join(chapters_dir, cf)
            with open(filepath, "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read() + "\n\n")
    log_status(f"Updated {output_file} with {len(chap_files)} chapter(s).")


# ==========================================
# PHASE 1: LOAD YAML
# ==========================================


def load_yaml(yaml_filename):
    print_header("Phase 1: Loading Book Structure")

    if not os.path.exists(yaml_filename):
        log_status(f"ERROR: '{yaml_filename}' not found.")
        log_status("Usage: python builder.py <yaml_file> [--model <model_name>]")
        sys.exit(1)

    log_status(f"Reading {yaml_filename}...")
    try:
        with open(yaml_filename, "r", encoding="utf-8") as f:
            book_state["yaml_data"] = yaml.safe_load(f)

        if not book_state["yaml_data"] or "chapters" not in book_state["yaml_data"]:
            log_status("ERROR: YAML file missing 'chapters' list.")
            sys.exit(1)

        chap_count = len(book_state["yaml_data"]["chapters"])
        log_status(f"Successfully loaded {chap_count} chapters.")

        # Show detected global metadata
        root = book_state["yaml_data"]
        meta = root.get("metadata_defaults", {})
        log_status(f"Title: {root.get('title') or meta.get('title', 'Untitled')}")
        log_status(
            f"Voice: {root.get('voice') or root.get('narrative_voice') or meta.get('narrative_voice', 'N/A')}"
        )
        log_status(f"Tone: {root.get('tone') or meta.get('tone', 'N/A')}")

    except Exception as e:
        log_status(f"ERROR: Failed to parse YAML: {e}")
        sys.exit(1)


# ==========================================
# PHASE 2: DRAFT CHAPTERS
# ==========================================


def phase_drafting():
    print_header("Phase 2: Drafting Chapters")

    chapters = book_state["yaml_data"].get("chapters", [])
    total_chapters = len(chapters)
    root_data = book_state["yaml_data"]

    log_status(f"Starting generation for {total_chapters} chapters...")

    for i, chap in enumerate(chapters):
        chap_num = chap.get("chapter_number", i + 1)
        current_progress = f"{i + 1}/{total_chapters}"
        chap_title = chap.get("title", "Untitled")

        filled = int((i / total_chapters) * 20)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"\n[PROGRESS] [{bar}] {current_progress} Complete")
        log_status(f"Processing Chapter {chap_num}: '{chap_title}'")

        if chap_num in book_state["chapters_content"]:
            log_status(f"Chapter {chap_num} already in memory. Skipping.")
            continue

        prev_sum = (
            chapters[i - 1].get("summary", "None (Start of book)")
            if i > 0
            else "None (Start of book)"
        )
        next_sum = (
            chapters[i + 1].get("summary", "None (End of book)")
            if i < len(chapters) - 1
            else "None (End of book)"
        )

        # DYNAMIC PROMPT GENERATION
        prompt = build_dynamic_prompt(chap, root_data, prev_sum, next_sum)

        try:
            content, usage, duration = make_llm_call(
                prompt,
                temperature=0.7,
                max_tokens=3000,
                context_label=f"Ch {chap_num} Draft",
            )
            book_state["chapters_content"][chap_num] = content
            log_status(
                f"Chapter {chap_num} drafted successfully. ({len(content)} chars)"
            )
            print(f"    {format_usage_line(usage, duration)}")

            # IMMEDIATE INCREMENTAL SAVE
            save_chapter_to_disk(chap_num, content, chap_title)

        except Exception as e:
            log_status(f"FAILED to draft Chapter {chap_num}. Error: {e}")
            output_dir = get_output_dir()
            log_status(
                f"Stopping execution. Check '{output_dir}/chapters/' and '{output_dir}/final_book.md' for recovered work."
            )
            sys.exit(1)

    log_status("All chapters drafted successfully.")


# ==========================================
# MAIN EXECUTION
# ==========================================


def main():
    parser = argparse.ArgumentParser(description="AI Book Architect")
    parser.add_argument(
        "yaml_file",
        nargs="?",
        help="Path to a TOC YAML file (default: example_toc.yaml)",
    )
    parser.add_argument("--model", help="Ollama model name")
    args = parser.parse_args()

    global MODEL_NAME
    if args.model:
        MODEL_NAME = args.model

    yaml_filename = args.yaml_file or "example_toc.yaml"
    if not os.path.exists(yaml_filename):
        log_status(f"ERROR: TOC file not found: {yaml_filename}")
        log_status("Usage: python builder.py <toc.yaml> [--model <model_name>]")
        sys.exit(1)

    print(f"*** PROCESS ID (PID): {os.getpid()} ***")
    print("Keep this PID handy in case the script becomes unresponsive.\n")

    log_status("Checking Ollama for available models...")

    print("-" * 50)
    print("Welcome to AI Book Architect (Dynamic YAML Mode)")
    print("Prompts are built automatically from your YAML structure.")
    print("Chapters are saved incrementally to prevent data loss.\n")

    load_yaml(yaml_filename)
    input("\nPress Enter to begin drafting chapters...")

    phase_drafting()

    print("\n✅ PROCESS COMPLETE.")
    output_dir = get_output_dir()
    print(f"Individual chapters: ./{output_dir}/chapters/")
    print(f"Full compiled book: ./{output_dir}/final_book.md")


if __name__ == "__main__":
    main()
