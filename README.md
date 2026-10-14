# BookBuilder (simple)

A small, readable two-stage pipeline that turns a **YAML table of contents** into
a **finished multi-chapter book** using a local LLM. No cloud account, no API
bill, no framework — just Python scripts and a plain-text outline you can edit by
hand.

The idea is simple: you write the structure, you pick the voice, the model writes
the prose. A built-in registry of **eleven author personas** — Hemingway's
iceberg, Didion's cold incision, McCarthy's biblical weight, Woolf's flowing
interior — injects a full literary style into every chapter prompt, so the same
outline can come back as a hard-boiled noir, a spare journalistic record, or a
stream-of-consciousness meditation.

```
 book_metadata.yaml  ──►  toc-maker.py  ──►  book_outline.yaml
  (your premise)           (outline stage)      (a TOC, expanded)
                                                       │
              writing_styles.py ──► a voice ───────────┤
              (11 author personas)                     ▼
                                             builder.py  ──►  output/<book>/chapters/chapter_001.md …
                                             (prose stage)     output/<book>/final_book.md
```

---

## What's in here

| Path | What it is |
|---|---|
| `tui.py` | **The recommended interface.** A Textual TUI: pick a TOC from a dropdown, press Generate, watch every chapter write live (tokens in / out / reasoning + time), and read the result as rendered markdown. Output goes to `output/<toc_name>/`. |
| `builder.py` | **Stage 2, interactive.** Reads a TOC YAML, writes one prompt per chapter, calls the model, saves each chapter to disk immediately, and recompiles `final_book.md` after every chapter. |
| `generate_book.py` | **Stage 2, headless.** The same job as `builder.py`, built for unattended runs: streaming, resumable, atomic writes, retry logic, and per-chapter token/time telemetry. It imports `builder.py`'s real prompt builder, so the prompts are identical. |
| `toc-maker.py` | **Stage 1.** Reads `book_metadata.yaml`, asks the model for a chapter outline, then expands each chapter with beats, then audits the result for continuity gaps. Resumable — it saves `generation_state` and picks up where it stopped. |
| `create_toc.py` | **Stage 1, one-shot alternative.** A single LLM call (Pydantic structured output) that writes a TOC in one go — no multi-phase approval. Note: its chapter fields (`number`, `length_words`) differ slightly from what Stage 2 reads (`chapter_number`, `word_count`), so rename them before drafting. |
| `writing_styles.py` | **The style registry — the feature that lets one outline become many books.** Eleven author personas (Austen, Chandler, Didion, Faulkner, Fitzgerald, Hemingway, McCarthy, Murakami, Thompson, Vonnegut, Woolf), each a voice profile plus hard execution rules, plus the `resolve_style` / `persona_block` helpers that inject a voice into the prompt. See [Literary styles](#literary-styles). |
| `example_toc.yaml` | **A complete, working example TOC** for a 10-chapter historical novel. This is the file to read first. |
| `the_vinland_saga/` | The book that `example_toc.yaml` produced, with the unedited model output. |
| `toc/default.YAML` | A minimal 2-chapter TOC — the smallest thing that runs. |
| `book_metadata.yaml` | Example premise file that feeds Stage 1. |
| `toc_type/*.yaml` | Optional genre "skeletons" (educational, fantasy, mystery, thriller). Reference material for shaping a TOC; not required by either script. |
| `NotNeeded/` | Scratch, logs and unused leftovers moved out of the way. Nothing here is required to run the pipeline. |

---

## Quickstart

Requires Python 3.10+, `pyyaml`, and a local model server
([LM Studio](https://lmstudio.ai) or [Ollama](https://ollama.com)). The TUI
additionally needs `textual` (`pip install textual`).

### 1. Configure the endpoint

Endpoint settings live in a `.env` file, never in the source. Copy the template
and fill it in — `.env.example` shows the exact format:

```powershell
Copy-Item .env.example .env     # Windows PowerShell
cp .env.example .env            # macOS / Linux
```

The file takes three settings:

| Setting | Purpose | Default |
|---|---|---|
| `LLM_BASE_URL` | OpenAI-compatible endpoint. LM Studio uses port 1234; Ollama uses 11434. | `http://localhost:1234/v1` |
| `LLM_MODEL` | Model name exactly as the server reports it (see `GET /v1/models`). | `gemma-4-12b-it` |
| `LLM_API_KEY` | Credential, sent as a bearer token when set. Optional for local servers. | none |

`.env` is git-ignored and must never be committed. Real environment variables
override it, so a shell export or a CI secret works too. A local LM Studio or
Ollama server does not check the credential, so leaving it blank is fine there.

An optional fourth setting, `LLM_REASONING_ALLOWANCE`, tunes the token headroom
reserved for a thinking model's scratchpad — see "Thinking models" below.

### Recommended: the TUI

With the endpoint configured, the whole pipeline is one command with a graphical
interface — no flags to remember:

```bash
python tui.py
```

- Pick a TOC from the dropdown (every valid `chapters:` YAML in the repo is listed).
- Pick a **style** from the dropdown next to it — "TOC default" defers to whatever
  the YAML says, or force a voice like *Joan Didion (didion)* for the whole run.
- Press **Generate** (or `g`). Watch each chapter announce itself, stream its
  character count, then report `in / out / reasoning / total` tokens and the time.
- Finished chapters appear in the file list on the right and are **rendered as
  markdown** — headings, bold and lists shown formatted, not as raw source.
- Output lands in `output/<toc_name>/` (e.g. `output/example_toc/final_book.md`).

Press `s` to stop (takes effect after the current chapter) and `q` to quit. The
next two steps drive the same pipeline from the command line.

### 2. Create a TOC (Stage 1)

Two ways. The multi-phase one expands and audits the outline, pausing so you can
review/edit it in `toc_review.yaml`:

```bash
python toc-maker.py          # reads book_metadata.yaml -> writes book_outline.yaml
```

Or the one-shot alternative (single LLM call, no approval loop):

```bash
python create_toc.py --title "My Book" --genre "Fantasy" --num-chapters 12 --target-words 80000
```

Either way you end up with a `chapters:` list you can hand-edit. (Skip this stage
entirely by writing a TOC by hand, or just use the bundled `example_toc.yaml`.)

### 3. Draft the book (Stage 2)

```bash
# Run the bundled example (The Vinland Saga) — the one-command way:
python generate_book.py --toc example_toc.yaml

# Or point it at any TOC YAML that has a top-level `chapters:` list:
python generate_book.py --toc path/to/your_toc.yaml

# Or the interactive original:
python builder.py example_toc.yaml
```

Both print the tokens used and the time taken for every chapter. Output lands in a
folder named after the book's title:

```
the_vinland_saga/
├── chapters/
│   ├── chapter_001.md
│   └── …
└── final_book.md
```

### Useful flags

```bash
python generate_book.py --dry-run            # build prompts, send nothing
python generate_book.py --only 3             # regenerate just chapter 3
python generate_book.py --only 3 --overwrite # force chapter 3 to be redone
python generate_book.py --model other-model  # different model for one run
python generate_book.py --style didion       # write the whole run in one voice
python generate_book.py --verbose            # stream characters as they arrive
```

An interrupted run resumes where it left off: any chapter already on disk is
skipped unless you pass `--overwrite`.

---

## The TOC format

The contract is deliberately tiny. **`chapters` is the only key that must exist.**
Everything else is optional, and *any* extra key you invent is rendered straight
into the prompt — which is what makes the format easy to extend.

```yaml
title: "The Vinland Saga"                       # optional
style: "hemingway"                              # optional — author persona (see Literary styles)
voice: "Third-person limited, past tense"       # optional — POV/narration, or a preset name
tone: "Grounded, wintry, pragmatic"             # optional

# Optional. Appended to every prompt as "AUTHOR GUIDELINES".
ai_instructions:
  global_rules:
    - "Keep the prose concrete."
    - "Avoid anachronism in food, tools and faith."

chapters:                                       # REQUIRED
  - chapter_number: 1
    title: "Timber and Salt"
    pov: "Hallvard, ship's steward"
    setting: "Brusand, Greenland"
    purpose: "Establish the crew and the decision to sail west"
    summary: >
      Hallvard tallies a warehouse that should not be empty…
    beats:
      - "Hallvard counts empty timber racks and lies about it in the ledger"
      - "Eirik produces a water-stained chart he refuses to explain"
    word_count: 1500
```

Five details worth knowing:

- **`style:` picks the voice.** Set it once at the root for the whole book, or on
  any single chapter to change voice for just that chapter. The value is a preset
  key (`hemingway`, `didion`, `mccarthy`, …) — see
  [Literary styles](#literary-styles).
- **`summary` chains the book.** Each chapter's prompt receives the summary of
  the previous *and* next chapter, so writing summaries as a continuous thread is
  what keeps the transitions seamless.
- **`word_count` sets the budget.** `generate_book.py` derives the token ceiling
  from it, plus an allowance for a thinking model's scratchpad.
- **Invention is free.** Adding `key_props`, `arcs_advanced`, `historical_basis`
  or anything else costs nothing — the prompt builder prints every key it finds as
  `Key Name: value`.
- **Two generations of the format exist.** The current *V2* shape is what both
  scripts use: book settings at the root, chapters under `chapters:`. Older *V1*
  files nested settings under `metadata_defaults:` and the chapter list under
  `toc:`. `builder.py` still reads V1 as a fallback, but new outlines should be
  written as V2. Stage 1 output (`book_outline.yaml`) is V1-shaped because it is an
  intermediate state file, not a hand-written TOC.

See `example_toc.yaml` for a fully annotated version of this format.

---

## Literary styles

This is the feature that lets one outline produce a completely different book.
`writing_styles.py` is a registry of author personas; pick one and the model
writes in that voice. The persona is injected into every chapter prompt under
`--- NARRATIVE VOICE & STYLISTIC PERSONA ---`, right above the chapter
assignment, so it steers the whole run without you touching the TOC.

### The eleven presets

Each preset is a voice profile (`talks_like`) plus a set of hard rules the model
must obey (`style_rules`). The keys are what you pass to `--style` or put in the
YAML.

| Key | Author | The voice |
|---|---|---|
| `austen` | Jane Austen | Witty ironic free indirect discourse. Manners as a battlefield; what is said and what is meant always differ. |
| `chandler` | Raymond Chandler | First-person hard-boiled cynic. Snappy dialogue, street-smart similes, vulnerability cracking through the armor. |
| `didion` | Joan Didion | Cool surgical incision. Rhythmic repetition, detached dread, clarity so sharp it devastates. |
| `faulkner` | William Faulkner | Dense winding stream-of-consciousness. Generational gravity; concrete nouns chained into long unbroken sentences. |
| `fitzgerald` | F. Scott Fitzgerald | Gilded jazz-age cadence. Romantic longing, shimmering atmosphere, gorgeous doomed yearning. |
| `hemingway` | Ernest Hemingway | The Iceberg Theory. Short declarative sentences, no adverbs, emotion only through action, 90% unspoken. |
| `mccarthy` | Cormac McCarthy | Sparse biblical weight. No quotation marks, polysyndeton chains of "and", raw physical materials, moral gravity. |
| `murakami` | Haruki Murakami | Flat detached melancholy. Hyper-specific mundane ritual, lonely orbital intimacy, dream-logic seeping in. |
| `thompson` | Hunter S. Thompson | Manic gonzo first person. Capitalized FEAR and INSTINCT, accelerating prose, someone hanging on at top speed. |
| `vonnegut` | Kurt Vonnegut | Deadpan conversational fatalism. Short punchy paragraphs, absurd comedy, "so it goes" shrugs. |
| `woolf` | Virginia Woolf | Fluid impressionistic stream-of-consciousness. Perception rippling into memory; sentences moving like water. |

### What actually gets injected

Run `--style didion` and every chapter prompt opens with a block like this
(text straight from `writing_styles.py`, wrapped here for the page):

```
--- NARRATIVE VOICE & STYLISTIC PERSONA ---
Emulate the authorial voice of: Joan Didion
Voice Profile: Cool precise incision. Rhythmic repetition hammering. Detached
California dread. Self-excavating clarity. The center will not hold. Surgical
observation of reality falling apart.

CRITICAL EXECUTION RULES:
Write with cool, surgical precision. Use rhythmic repetition—phrases that loop
and return like a nervous tic, building quiet hypnotic tension. The narrator
observes events and emotions from a clinical, forensic distance. Avoid overt
warmth; deliver clarity so sharp it becomes devastating. Every sentence a clean
incision.
```

### Picking a style

Four ways, in order of precedence:

| # | Where | How |
|---|---|---|
| 1 | **CLI** | `--style didion` — overrides everything for one run |
| 2 | **Chapter YAML** | `style: chandler` on a single chapter |
| 3 | **Root YAML** | `style: hemingway` at the top — the whole book |
| 4 | **TUI** | the style dropdown ("TOC default" defers to the YAML) |

```bash
# one run, one voice — no TOC edits needed
python generate_book.py --toc example_toc.yaml --style mccarthy
```

```yaml
# or bake it into the TOC — once for the whole book…
style: hemingway
chapters:
  - chapter_number: 1
    title: "Timber and Salt"
    …
```

### Mixing voices

Because `style:` is per-chapter, a single book can change voice on purpose. A
noir interrogation and a lyrical interior can sit side by side:

```yaml
chapters:
  - chapter_number: 7
    title: "The Warehouse"
    style: chandler        # hard-boiled, first-person, cynical
    …
  - chapter_number: 8
    title: "After"
    style: woolf           # stream-of-consciousness — the aftermath felt inward
    …
```

### How resolution actually works

`resolve_style` walks these candidates **in order and takes the first that maps
to a preset**:

1. the CLI `--style` — authoritative: if it matches a preset you get that persona,
   otherwise it is passed straight through as a free-form voice string,
2. the chapter's `style:`,
3. the root's `style:`,
4. the chapter's `voice:` — *only if it names a preset*,
5. the root's `voice:` — *only if it names a preset*,
6. the root's `narrative_voice:` — *only if it names a preset*,
7. `metadata_defaults.narrative_voice:` — *only if it names a preset*,

and otherwise falls back to the first non-empty value as a free-form `Voice:`
line in the prompt.

Step 4 is the subtle one. `voice:` is a common field for a POV character
(`voice: "Hallvard, ship's steward"`). If any bare string were allowed to win,
that character name would silently disable a real `style:` set elsewhere. So a
`voice:` only participates in style resolution when it actually matches a preset
name; a POV string falls through to the free-form line and never shadows the
persona.

### Adding your own

Append a `StylePreset` to `PRESETS` in `writing_styles.py` — nothing else to wire
up, and `--style calvino` (or `style: calvino` in a TOC) works immediately:

```python
PRESETS["calvino"] = StylePreset(
    name="calvino",
    author="Italo Calvino",
    talks_like=(
        "Light, crystalline, playful. Combinatorial imagination. "
        "Precise logical whimsy. Cities described as thought experiments."
    ),
    style_rules=(
        "Write with crystalline clarity and playful precision. Let the "
        "imagination run combinatorial—describe the impossible as if it were "
        "an engineering problem. Keep the surface light; the ideas do the work."
    ),
)
```

---

## What the telemetry tells you

`generate_book.py` prints a card after every chapter, using the token counts the
server reports rather than estimates:

```
  ==============================================================
  CHAPTER 10  ·  Homeward Is Not the Same
  ==============================================================
  OUTPUT
  Prose written              1,183 words
    output tokens            1,522
    chars per token          4.2
  Reasoning chars            3,330
  --------------------------------------------------------------
  TOKENS
  Prompt tokens              1,857          input
  Completion tokens          2,393          all output
    reasoning portion        871            scratchpad
    visible prose portion    1,522          the writing
  Total tokens               4,250
  --------------------------------------------------------------
  TIME
  Total wall clock           59.0s
  Time to first token        28.55s
  Generation time            30.5s          first token to end
  Throughput (all out)       102.9 tok/s    incl. scratchpad
  Throughput (prose)         53.0 tok/s     visible text only
  Prose rate                 1295 words/min wall clock
```

A run summary and totals follow at the end. If the server sends no usage block,
the fields read `n/a` rather than being invented.

---

## Thinking models

Some models emit a hidden reasoning channel next to the visible answer. They work
fine — they just need a token budget large enough for the scratchpad **and** the
prose. Given too small a budget, a thinking model spends the whole allotment
reasoning and returns no visible text at all, which looks exactly like a broken
model or a silent failure but is purely a budgeting problem.

`generate_book.py` handles this by reserving extra tokens on top of the prose
budget for every chapter. Tune it with `LLM_REASONING_ALLOWANCE` in `.env`, or
override the total with `--max-tokens`. The reasoning and prose token counts are
reported separately so you can see exactly how much budget the scratchpad is
consuming. A non-thinking model simply leaves the allowance unused.

---

## Notes and limitations

- **It's a first draft machine, not a finished-book machine.** Chapter lengths
  vary, and continuity drifts across a long book. `toc-maker.py` has an audit pass
  that flags weak transitions and can patch chapters afterwards; that's the
  intended fix, not manual editing alone.
- **The stage-1 metadata file is just a starting point.** `book_metadata.yaml` is a
  sample premise; rewrite it for your own book, or skip stage 1 entirely and write
  a TOC by hand.
- **Both stage-2 scripts tell you the file path they were looking for** if the TOC
  YAML does not exist.

---

## About the example book

`example_toc.yaml` outlines *The Vinland Saga* — a fictional Norse timber voyage
that makes landfall in North America around 1010 CE, and the winter the crew
spends among people whose language and customs they do not share.

It is loosely inspired by the Norse voyages described in the Vinland Sagas and by
the archaeological site at L'Anse aux Meadows, Newfoundland. The characters,
events and settlement details are invented. The prose in `the_vinland_saga/` is
unedited model output from `gemma-4-12b-it`, included so you can judge the
pipeline's actual quality rather than a polished sample.
