#!/usr/bin/env python3
"""
Genre-Agnostic LLM Book Outline Generator
Phases: TOC → Approve → Expand → Audit → Patch
Metadata-driven, resume-safe, single-file CLI.
LLM outputs JSON for reliability. State saved as YAML.
"""

import json
import os
import re
import sys
import time

import yaml
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE = "book_metadata.yaml"
OUTPUT_FILE = "book_outline.yaml"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma-4-12b-it-aeon-32k:latest")
CLIENT = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

USE_JSON_OUTPUT = True  # LLM outputs JSON. Python converts to YAML for storage.

STRUCTURAL_KEYS = {
    "chapters",
    "toc",
    "generation_state",
    "audit_results",
    "num_chapters",
    "output_file",
    "input_file",
}


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING & UTILS
# ─────────────────────────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def wait_for_enter(prompt_msg):
    try:
        input(f"\n{prompt_msg} (Press Enter to continue) ")
    except KeyboardInterrupt:
        log("Interrupted. Exiting.", "WARN")
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# I/O & PARSING
# ─────────────────────────────────────────────────────────────────────────────
def load_yaml(path):
    if not os.path.exists(path):
        log(f"File not found: {path}", "ERROR")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            indent=2,
            sort_keys=False,
            allow_unicode=True,
        )
    log(f"Saved state to {path}")


def normalize_llm_output(raw: str) -> str:
    """Fix common LLM formatting drift before parsing."""
    # Strip markdown
    raw = re.sub(r"^```(?:json|yaml)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    # Fix JSON trailing commas
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    # Fix missing list prefix for audit boundaries
    raw = re.sub(r"(?m)^(boundary_\d+_to_\d+:)", r"- \1", raw)
    return raw.strip()


def extract_json_block(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in response")
    return raw[start : end + 1]


def parse_llm_response(raw, max_retries=2, prompt_fn=None, **prompt_kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            cleaned = normalize_llm_output(raw)
            if USE_JSON_OUTPUT:
                parsed = json.loads(extract_json_block(cleaned))
            else:
                parsed = yaml.safe_load(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a mapping")
            return parsed
        except Exception as e:
            log(f"Parse failed (attempt {attempt}): {e}", "WARN")
            if attempt == max_retries or not prompt_fn:
                with open("llm_raw_response.txt", "w", encoding="utf-8") as f:
                    f.write(raw)
                log("Raw LLM output saved to llm_raw_response.txt", "WARN")
                raise RuntimeError(
                    f"Failed to parse LLM output after {max_retries} attempts"
                ) from e
            log("Retrying with stricter formatting instructions...", "INFO")
            strict_prompt = (
                prompt_fn(**prompt_kwargs)
                + "\n\nCRITICAL: Output ONLY valid JSON. No markdown, no explanations, no trailing commas."
            )
            raw = llm_call(strict_prompt, temperature=0.2, context="Parse Retry")


# ─────────────────────────────────────────────────────────────────────────────
# METADATA FORMATTING
# ─────────────────────────────────────────────────────────────────────────────
def format_dynamic_metadata(root_data):
    lines = []
    for key, value in root_data.items():
        if key in STRUCTURAL_KEYS or value is None:
            continue
        if isinstance(value, (list, dict)):
            lines.append(f"{key}:")
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            lines.append(f"  - {k}: {v}")
                    else:
                        lines.append(f"  - {item}")
            else:
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
STRICT_SCHEMA_TOC = """
REQUIRED JSON STRUCTURE:
{
  "toc": [
    {
      "chapter_number": int,
      "title": "string",
      "summary": "string",
      "pacing_phase": "string",
      "intensity_level": int,
      "key_beats": ["string", "string", "string"],
      "genre_specific": {"metadata_key": "value"}
    }
  ]
}
Do NOT add, rename, or nest fields differently.
"""

STRICT_SCHEMA_EXPANSION = """
REQUIRED JSON STRUCTURE:
{
  "chapter": {
    "chapter_number": int,
    "title": "string",
    "summary": "string",
    "pacing_phase": "string",
    "intensity_level": int,
    "key_beats": [{"beat": "string", "scene": "string", "direction": "string", "reader_payoff": "string", "continuity_hook": "string"}],
    "genre_specific": {},
    "tone_calibration": "string",
    "continuity_hooks": [{"hook": "string", "chapter": int}]
  }
}
Preserve existing fields. Expand only. Do NOT invent new top-level keys.
"""


def build_toc_prompt(metadata, num_to_generate, start_num):
    meta_block = format_dynamic_metadata(metadata)
    return f"""You are a developmental editor structuring a book outline.

BOOK METADATA:
{meta_block}

Generate {num_to_generate} chapter outlines. Start numbering at {start_num}.

{STRICT_SCHEMA_TOC}

RULES:
- Output ONLY valid JSON matching the exact structure above.
- key_beats MUST be a list of strings. genre_specific MUST be a flat dict using ONLY provided metadata keys.
- No markdown, no explanations, no conversational text.
"""


def build_expansion_prompt(metadata, toc_entry, pass_num):
    meta_block = format_dynamic_metadata(metadata)
    chapter_ctx = yaml.dump(toc_entry, default_flow_style=False, indent=2)
    focus = (
        "structure, scene breakdown, character focus, and metadata alignment"
        if pass_num == 1
        else "pacing directives, reader payoff, continuity hooks, and tone calibration"
    )
    return f"""You are expanding a chapter outline. This is pass {pass_num}.

BOOK METADATA:
{meta_block}

CHAPTER TO EXPAND:
{chapter_ctx}

INSTRUCTIONS:
- Deepen the outline. Add {focus}.
- Keep all existing fields unchanged.
- Expand genre_specific with detailed, contextual mappings to the provided metadata.
{STRICT_SCHEMA_EXPANSION}
- Format: JSON only. No markdown, no explanations.
"""


def build_audit_rubric(metadata):
    rubric = [
        "Pacing/Structure Alignment: Chapters must logically follow or distribute any pacing_structure, arc, or progression metadata provided.",
        "Intensity Flow: intensity_level should progress naturally. Allow ±1 jumps at pacing_phase transitions (e.g., Setup→Buildup). Flag unexplained spikes/drops >2 points, or any regression within the same phase.",
        "Chapter Transitions: Check end of Ch N → start of Ch N+1. Flag jarring tone shifts, dropped threads, or missing narrative hooks.",
        "Metadata Consistency: genre_specific fields should escalate, rotate, or resolve logically. Flag random repetition or unexplained disappearance.",
        "Structural Integrity: Every chapter must have chapter_number, title, summary, pacing_phase, intensity_level, key_beats, and genre_specific.",
    ]
    arc_keys = [k for k in metadata if "arc" in k.lower() or "progression" in k.lower()]
    if arc_keys:
        rubric.append(
            f"Arc Tracking: Monitor progression of {', '.join(arc_keys)}. Ensure logical escalation → peak → resolution."
        )
    param_keys = [
        k
        for k in metadata
        if any(x in k.lower() for x in ["param", "element", "type", "focus"])
    ]
    if param_keys:
        rubric.append(
            f"Parameter Distribution: {', '.join(param_keys)} should be introduced, developed, and paid off intentionally."
        )
    return "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rubric))


def build_audit_prompt(metadata, chapter_chunk):
    rubric = build_audit_rubric(metadata)
    meta_block = format_dynamic_metadata(metadata)
    chapters_block = yaml.dump(chapter_chunk, default_flow_style=False, indent=2)
    return f"""You are a developmental editor auditing chapter flow, pacing, and metadata consistency.

BOOK METADATA:
{meta_block}

CHAPTERS TO REVIEW:
{chapters_block}

AUDIT RUBRIC:
{rubric}

OUTPUT FORMAT (JSON ONLY):
{{
  "review": [
    {{
      "chapter_number": 1,
      "flags": ["Issue 1", "Issue 2"],
      "suggestions": ["Fix 1", "Fix 2"]
    }},
    {{
      "boundary_1_to_2": {{
        "transition_quality": "smooth | jarring | needs_hook",
        "missing_threads": ["Thread 1"],
        "recommended_bridge": "1-sentence suggestion"
      }}
    }}
  ],
  "global_notes": "1-2 sentences on overall arc health"
}}

Respond with valid JSON only. If a chapter passes, use "flags": ["PASS"]
"""


def build_patch_prompt(metadata, chapter_data, flags):
    meta_block = format_dynamic_metadata(metadata)
    ch_block = yaml.dump(chapter_data, default_flow_style=False, indent=2)
    flags_str = "\n".join(f"- {f}" for f in flags)
    return f"""Revise the following chapter outline to address these audit flags:
{flags_str}

BOOK METADATA:
{meta_block}

CURRENT CHAPTER:
{ch_block}

RULES:
- Fix only the flagged issues. Preserve all other fields.
- Keep genre_specific consistent with metadata.
{STRICT_SCHEMA_EXPANSION}
- Format: JSON only. No explanations.
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def llm_call(prompt, temperature=0.7, max_tokens=4000, context="LLM"):
    log(f"{context}: Sending request...", "INFO")
    start = time.time()
    try:
        resp = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        duration = round(time.time() - start, 2)
        content = resp.choices[0].message.content.strip()
        log(f"{context}: Received response in {duration}s.", "INFO")
        return content
    except Exception as e:
        log(f"{context}: ERROR during LLM call: {e}", "ERROR")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: TOC GENERATION & APPROVAL
# ─────────────────────────────────────────────────────────────────────────────
def phase1_generate_toc(state):
    log("PHASE 1: Generating lightweight TOC...", "INFO")
    meta = state
    target = meta.get("num_chapters", 10)
    existing = len(meta.get("toc", []))
    if existing >= target:
        log("TOC already meets target chapter count.", "INFO")
        state.setdefault("generation_state", {})["phase"] = "toc_approved"
        return state

    num_to_gen = target - existing
    start_num = existing + 1
    prompt = build_toc_prompt(meta, num_to_gen, start_num)
    raw = llm_call(prompt, temperature=0.6, context="TOC Gen")
    parsed = parse_llm_response(
        raw,
        prompt_fn=build_toc_prompt,
        metadata=meta,
        num_to_generate=num_to_gen,
        start_num=start_num,
    )

    new_toc = parsed.get("toc", [])
    if not new_toc:
        raise RuntimeError("LLM returned empty TOC")

    state.setdefault("toc", []).extend(new_toc)
    state.setdefault("generation_state", {})["phase"] = "toc_generated"
    save_yaml(state, OUTPUT_FILE)
    log(f"Generated {len(new_toc)} TOC entries.", "INFO")
    return state


def phase1_approve_toc(state):
    log("PHASE 1.5: Awaiting TOC approval...", "INFO")
    toc_file = "toc_review.yaml"
    save_yaml({"toc": state["toc"]}, toc_file)
    log(f"TOC saved to {toc_file}. Edit if needed.", "INFO")
    wait_for_enter("Review the TOC. Make any manual edits to the file if desired.")

    reviewed = load_yaml(toc_file)
    state["toc"] = reviewed.get("toc", state["toc"])
    state["generation_state"]["phase"] = "toc_approved"
    save_yaml(state, OUTPUT_FILE)
    log("TOC approved.", "INFO")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: PER-CHAPTER EXPANSION
# ─────────────────────────────────────────────────────────────────────────────
def phase2_expand_chapters(state, max_passes=2):
    log("PHASE 2: Expanding chapters iteratively...", "INFO")
    gs = state.setdefault("generation_state", {})
    last_idx = gs.get("last_expanded_chapter", 0)
    current_pass = gs.get("current_pass", 1)
    toc = state.get("toc", [])
    chapters = state.setdefault("chapters", [])

    while len(chapters) < len(toc):
        chapters.append({"chapter_number": len(chapters) + 1, "status": "pending"})

    for i in range(last_idx, len(toc)):
        toc_entry = toc[i]
        ch_num = toc_entry["chapter_number"]
        log(f"Expanding Chapter {ch_num} (Pass {current_pass})...", "INFO")

        prompt = build_expansion_prompt(state, toc_entry, current_pass)
        raw = llm_call(
            prompt, temperature=0.7, context=f"Ch{ch_num} Pass{current_pass}"
        )
        parsed = parse_llm_response(
            raw,
            prompt_fn=build_expansion_prompt,
            metadata=state,
            toc_entry=toc_entry,
            pass_num=current_pass,
        )

        expanded = parsed.get("chapter", {})
        chapters[i].update(expanded)
        chapters[i]["status"] = f"pass{current_pass}_complete"

        gs["last_expanded_chapter"] = i + 1
        save_yaml(state, OUTPUT_FILE)

    if current_pass < max_passes:
        gs["current_pass"] = current_pass + 1
        gs["last_expanded_chapter"] = 0
        log(
            f"Pass {current_pass} complete. Starting Pass {current_pass + 1}...", "INFO"
        )
        return phase2_expand_chapters(state, max_passes)

    gs["phase"] = "expansion_complete"
    save_yaml(state, OUTPUT_FILE)
    log("Phase 2 complete.", "INFO")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: CONTINUITY & PACING AUDIT
# ─────────────────────────────────────────────────────────────────────────────
def phase3_audit_flow(state):
    log("PHASE 3: Auditing pacing & continuity...", "INFO")
    chapters = state.get("chapters", [])
    if len(chapters) < 2:
        log("Not enough chapters to audit.", "WARN")
        state["generation_state"]["phase"] = "audit_complete"
        return state

    window_size = 3
    step = window_size - 1
    all_reviews = []

    for i in range(0, len(chapters) - 1, step):
        chunk = chapters[i : i + window_size]
        prompt = build_audit_prompt(state, chunk)
        raw = llm_call(prompt, temperature=0.3, context=f"Audit Window {i // step + 1}")
        parsed = parse_llm_response(
            raw, prompt_fn=build_audit_prompt, metadata=state, chapter_chunk=chunk
        )
        all_reviews.extend(parsed.get("review", []))

    seen = set()
    merged = []
    for r in all_reviews:
        key = str(r.get("chapter_number") or r.get("boundary_X_to_Y"))
        if key not in seen:
            seen.add(key)
            merged.append(r)

    state["audit_results"] = merged
    state["generation_state"]["phase"] = "audit_complete"
    save_yaml(state, OUTPUT_FILE)
    log(f"Audit complete. {len(merged)} review entries saved.", "INFO")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: TARGETED PATCHES (BATCH MODE)
# ─────────────────────────────────────────────────────────────────────────────
def phase4_patch_flagged(state):
    log("PHASE 4: Applying targeted patches...", "INFO")
    audits = state.get("audit_results", [])
    chapters = state.get("chapters", [])
    patched = state["generation_state"].setdefault("patched_chapters", [])

    # Collect only chapter flags (skip boundaries & PASS)
    flagged = [
        e
        for e in audits
        if e.get("chapter_number")
        and e.get("flags") not in (["PASS"], [], None)
        and e["chapter_number"] not in patched
    ]

    if not flagged:
        log("No chapters need patching.", "INFO")
        state["generation_state"]["phase"] = "complete"
        save_yaml(state, OUTPUT_FILE)
        return state

    print(
        f"\nFound {len(flagged)} flagged chapters: {[e['chapter_number'] for e in flagged]}"
    )
    resp = input("Apply patches? (y/n/a for all): ").strip().lower()
    if resp == "n":
        state["generation_state"]["phase"] = "complete"
        save_yaml(state, OUTPUT_FILE)
        return state

    for entry in flagged:
        ch_num = entry["chapter_number"]
        if resp != "a":
            if input(f"Patch Ch{ch_num}? (y/n): ").strip().lower() != "y":
                continue

        ch_data = next((c for c in chapters if c.get("chapter_number") == ch_num), {})
        if not ch_data:
            log(f"Chapter {ch_num} not found in state.", "WARN")
            continue

        prompt = build_patch_prompt(state, ch_data, entry["flags"])
        raw = llm_call(prompt, temperature=0.4, context=f"Patch Ch{ch_num}")
        parsed = parse_llm_response(
            raw,
            prompt_fn=build_patch_prompt,
            metadata=state,
            chapter_data=ch_data,
            flags=entry["flags"],
        )

        updated = parsed.get("chapter", {})
        ch_data.update(updated)
        ch_data["status"] = "patched"
        patched.append(ch_num)
        save_yaml(state, OUTPUT_FILE)
        log(f"Chapter {ch_num} patched.", "INFO")

    state["generation_state"]["phase"] = "complete"
    save_yaml(state, OUTPUT_FILE)
    log("Pipeline complete. Final outline saved.", "INFO")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────
def get_or_init_state():
    if os.path.exists(OUTPUT_FILE):
        log(f"Resuming from {OUTPUT_FILE}...", "INFO")
        return load_yaml(OUTPUT_FILE)

    log(f"Loading base metadata from {INPUT_FILE}...", "INFO")
    base = load_yaml(INPUT_FILE)
    base.setdefault(
        "generation_state",
        {
            "phase": "init",
            "last_expanded_chapter": 0,
            "current_pass": 1,
            "patched_chapters": [],
        },
    )
    save_yaml(base, OUTPUT_FILE)
    return base


def main():
    log("Starting Genre-Agnostic Book Outline Generator", "INFO")
    state = get_or_init_state()
    phase = state["generation_state"]["phase"]

    pipeline = [
        ("init", phase1_generate_toc),
        ("toc_generated", phase1_approve_toc),
        ("toc_approved", phase2_expand_chapters),
        ("expansion_complete", phase3_audit_flow),
        ("audit_complete", phase4_patch_flagged),
    ]

    start_idx = next((i for i, (p, _) in enumerate(pipeline) if p == phase), 0)

    for p_name, func in pipeline[start_idx:]:
        if state["generation_state"]["phase"] != p_name:
            continue
        try:
            state = func(state)
        except Exception as e:
            log(f"Pipeline failed at {p_name}: {e}", "ERROR")
            sys.exit(1)

    log(f"All phases complete. Output saved to {OUTPUT_FILE}", "INFO")


if __name__ == "__main__":
    main()
