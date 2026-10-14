"""writing_styles.py — Modular literary style presets for BookBuilder.

Each preset captures an author's cadence (`talks_like`) and the hard rules the
model must obey (`style_rules`). The prompt builder injects these into the
chapter flow so a run can be written in any of these voices.

Resolution priority (see `resolve_style`):
    CLI --style  >  chapter style:  >  root style:  >  chapter/root voice: (only
    if it matches a preset name)  >  freeform voice string.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StylePreset:
    name: str
    author: str
    talks_like: str
    style_rules: str


PRESETS: dict[str, StylePreset] = {
    "austen": StylePreset(
        name="austen",
        author="Jane Austen",
        talks_like=(
            "Witty ironic free indirect discourse. Precise social scalpel. "
            "Drawing-room warfare elegant. Manners as battlefield. Perfectly placed subordinate "
            "clauses. Arch knowing narrator. Civility papering over desire."
        ),
        style_rules=(
            "Use free indirect discourse—slipping between third-person narration and character "
            "internal thoughts without announcement. Maintain arch, ironic narrative distance: "
            "the narrator observes social comedy with clarity and gentle mockery. Dialogue must be "
            "precise, layered, and socially weaponized; what is said and what is meant are always "
            "different. Restrained, civilized, devastating."
        ),
    ),
    "chandler": StylePreset(
        name="chandler",
        author="Raymond Chandler",
        talks_like=(
            "First-person cynical monologue. Wisecracking tough guy. Snappy hard-boiled dialogue. "
            "Sharp street-smart similes. World-weary wit. Grim humor cracking through. "
            "Understated vulnerability beneath armor. Cigarette smoke and bourbon voice."
        ),
        style_rules=(
            "Write in the first-person internal monologue of a hard-boiled, cynical narrator. "
            "Use sharp, fast-paced dialogue and highly specific, gritty similes. Reveal emotion "
            "strictly through understated cracking of cynical armor. Every observation must feel "
            "like someone who has seen the worst of the world and is surprised by any decency."
        ),
    ),
    "didion": StylePreset(
        name="didion",
        author="Joan Didion",
        talks_like=(
            "Cool precise incision. Rhythmic repetition hammering. Detached California dread. "
            "Self-excavating clarity. The center will not hold. Surgical observation of reality "
            "falling apart."
        ),
        style_rules=(
            "Write with cool, surgical precision. Use rhythmic repetition—phrases that loop and "
            "return like a nervous tic, building quiet hypnotic tension. The narrator observes "
            "events and emotions from a clinical, forensic distance. Avoid overt warmth; deliver "
            "clarity so sharp it becomes devastating. Every sentence a clean incision."
        ),
    ),
    "faulkner": StylePreset(
        name="faulkner",
        author="William Faulkner",
        talks_like=(
            "Dense winding syntax. Stream-of-consciousness memory bleed. Heavy rhythmic weight. "
            "Concrete nouns chained into eternity. Generational gravity. Unbroken temporal flow."
        ),
        style_rules=(
            "Use signature long, winding sentences and stream-of-consciousness. Let internal "
            "thoughts bleed directly into physical observations. Ground long sentences strictly "
            "using concrete nouns, strong verbs, and physical actions—do not stack empty "
            "adjectives or abstract romantic metaphors. Tone must be heavy, sensory, and deeply "
            "anchored in time."
        ),
    ),
    "fitzgerald": StylePreset(
        name="fitzgerald",
        author="F. Scott Fitzgerald",
        talks_like=(
            "Elegant golden-age cadence. Jazz-age glamour shimmering. Beautiful restrained rhythms. "
            "Romantic longing pitched perfect. Moonlight and champagne syntax. Gorgeous doomed yearning. "
            "Every sentence a small chandelier."
        ),
        style_rules=(
            "Use elegant, beautifully cadenced prose with a sense of gilded glamour and deep "
            "poignancy. Graceful restraint is key. Settings must shimmer with atmospheric "
            "melancholy: light catching glass, distant music, a feeling of beautiful transience. "
            "Sentences must be finely wrought without tipping into purple melodrama."
        ),
    ),
    "hemingway": StylePreset(
        name="hemingway",
        author="Ernest Hemingway",
        talks_like=(
            "Short punchy declarative sentences. Spare concrete nouns. No adverbs. Zero fluff. "
            "Action reveals emotion. Dialogue real and clipped. Iceberg subtext deep underneath. "
            "Hard clean rhythm. What is not said matters more."
        ),
        style_rules=(
            "Apply the Iceberg Theory: characters may feel intense emotion, but must never declare "
            "it outright or explain it internally. Use short, declarative sentences. Strictly ban "
            "adverbs and abstract adjectives. Rely entirely on concrete physical actions to imply "
            "internal states. Dialogue must be clipped, leaving 90% unspoken."
        ),
    ),
    "mccarthy": StylePreset(
        name="mccarthy",
        author="Cormac McCarthy",
        talks_like=(
            "Sparse biblical cadence. Polysyndetic chains. No quotation marks ever. Bleak epic tone. "
            "Archaic rhythms. Concrete harsh materials. Terse moral weight. Old Testament gravity. "
            "Speech woven into prose. And clauses stacked and stacked."
        ),
        style_rules=(
            "Do not use quotation marks for dialogue; weave spoken lines directly into narrative "
            "prose. Use a sparse, biblical rhythm heavily reliant on 'and' (polysyndeton). "
            "Ground the scene in raw, physical elements (dust, rock, iron, smoke, cold) and omit "
            "modern psychological jargon entirely. Sentences must carry biblical moral weight."
        ),
    ),
    "murakami": StylePreset(
        name="murakami",
        author="Haruki Murakami",
        talks_like=(
            "Flat detached melancholic. Passive lonely narrator. Hyper-specific mundane details. "
            "Jazz record playing quietly. Clinical intimacy veiling deep longing. Quiet isolation "
            "humming. Dream logic surrealism seeping in. Spaghetti at midnight."
        ),
        style_rules=(
            "Maintain a flat, detached, slightly melancholic first-person narrator. Focus on "
            "hyper-specific mundane rituals (preparing simple meals, listening to specific records, "
            "the hum of an appliance). Intimacy and conflict should feel like lonely celestial bodies "
            "passing in orbit. Allow subtle dream-logic or ambient strangeness to seep into the room."
        ),
    ),
    "thompson": StylePreset(
        name="thompson",
        author="Hunter S. Thompson",
        talks_like=(
            "Manic frenetic paranoid cadence. Drug-fueled urgency. Capitalized FEAR and MADNESS. "
            "Sweating accelerating prose. Violent visceral energy. Gonzo first-person assault. "
            "Wild-eyed survivor narration. Pure adrenal INSTINCT."
        ),
        style_rules=(
            "Write in first person with a manic, accelerating cadence. Use strategic capitalization "
            "for dread and shock (e.g., The FEAR, Total CHAOS, Pure INSTINCT). Avoid flowery sentiment; "
            "compare mundane scenes to high-speed collisions, adrenaline rushes, or impending doom. "
            "Prose should read like someone hanging on to the steering wheel at top speed."
        ),
    ),
    "vonnegut": StylePreset(
        name="vonnegut",
        author="Kurt Vonnegut",
        talks_like=(
            "Deadpan conversational. Short paragraph punch. So it goes fatalism. Midwestern "
            "plainness. Absurd human comedy. Dry delivery. Everything was beautiful and nothing hurt."
        ),
        style_rules=(
            "Write in short, deceptively simple paragraphs. Deliver devastating observations with "
            "conversational, folksy deadpan. Embrace fatalistic humor: the universe is absurd and "
            "the only sane response is a shrug. Use recurring refrains sparingly. Avoid grand "
            "declarations; treat characters as vulnerable humans caught in larger machinery."
        ),
    ),
    "woolf": StylePreset(
        name="woolf",
        author="Virginia Woolf",
        talks_like=(
            "Fluid lyrical stream-of-consciousness. Impressionistic internal shifts. Rhythmic "
            "thought cascades. External dissolves into memory. Light luminous poetic. "
            "Fleeting perceptions captured mid-flight. Moment blooming into eternity."
        ),
        style_rules=(
            "Use fluid, impressionistic stream-of-consciousness where the external world dissolves "
            "into interior memory. Focus on the rhythm of perception: how a glance, a glass placed "
            "down, or a passing shadow triggers ripples of psychological reaction. Sentences must "
            "move like water—luminous, reflective, and constantly shifting between past and present."
        ),
    ),
}


def get_style_preset(key_or_name: str | None) -> StylePreset | None:
    """Look up a preset by key (case-insensitive). Returns None if unknown."""
    if not key_or_name:
        return None
    cleaned = key_or_name.strip().lower()
    return PRESETS.get(cleaned)


def resolve_style(
    chapter: dict,
    root: dict,
    style_override: str | None = None,
) -> tuple[StylePreset | None, str | None]:
    """
    Resolve the effective style for one chapter.

    Returns `(preset, raw)` where `preset` is a StylePreset when the chosen value
    matches a registry key, and `raw` is the winning string (used for the freeform
    `Voice:` line when no preset matched).

    Priority: CLI override > chapter style: > root style: > chapter/root voice:
    (only when it matches a preset) > freeform voice string.
    """
    metadata = root.get("metadata_defaults") or {}
    candidates = [
        chapter.get("style"),
        root.get("style"),
        chapter.get("voice"),
        root.get("voice"),
        root.get("narrative_voice"),
        metadata.get("narrative_voice"),
    ]

    # A CLI override is authoritative: use it as a preset if it matches, otherwise
    # treat it as a freeform voice string.
    if style_override:
        return get_style_preset(style_override), style_override

    # Otherwise take the first YAML value that actually maps to a preset, so a
    # POV-style `voice:` on one chapter cannot shadow a valid `style:` elsewhere.
    for cand in candidates:
        if cand:
            preset = get_style_preset(cand)
            if preset is not None:
                return preset, cand

    # No preset anywhere: fall back to the first non-empty value as freeform voice.
    raw = next((c for c in candidates if c), None)
    return None, raw


def persona_block(preset: StylePreset | None) -> list[str]:
    """The NARRATIVE VOICE block, or an empty list when no preset was chosen."""
    if preset is None:
        return []
    return [
        "--- NARRATIVE VOICE & STYLISTIC PERSONA ---",
        f"Emulate the authorial voice of: {preset.author}",
        f"Voice Profile: {preset.talks_like}",
        "",
        "CRITICAL EXECUTION RULES:",
        preset.style_rules,
        "",
    ]
