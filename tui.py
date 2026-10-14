#!/usr/bin/env python3
"""
tui.py — a Textual front-end for the BookBuilder pipeline.

Pick a TOC, press Generate, and watch it write the book live: every chapter
announces itself, streams its character count as it arrives, then reports the
tokens spent (input / output / reasoning / total) and the wall-clock time.
The finished chapters appear in a file list and are rendered as markdown on the
right.

It reuses generate_book.py as the engine, so the prompts, retry logic and output
format are byte-for-byte the same as the headless runner. This file only adds the
interface.

    python tui.py

Requires: textual (pip install textual)
"""

from __future__ import annotations

import os
import threading
import time

import yaml
from textual.app import App, ComposeResult
from textual import on, work
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Markdown,
    RichLog,
    Select,
    Static,
)

import generate_book as gb
from writing_styles import PRESETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_tocs(base: str) -> list[str]:
    """Find every YAML file under `base` that has a non-empty `chapters` list."""
    skip = {".git", ".venv", "NotNeeded", "output", "__pycache__", ".history-files"}
    found: list[str] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for name in files:
            if not name.lower().endswith((".yaml", ".yml")):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    data = yaml.safe_load(handle)
            except Exception:
                continue
            if isinstance(data, dict) and isinstance(data.get("chapters"), list) and data["chapters"]:
                found.append(path)
    return sorted(found)


# ---------------------------------------------------------------------------
# Messages passed from the worker thread to the UI
# ---------------------------------------------------------------------------


class LogMsg(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class StatusMsg(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class ChapterStarted(Message):
    def __init__(self, number: int, title: str, total: int) -> None:
        super().__init__()
        self.number = number
        self.title = title
        self.total = total


class ChapterProgress(Message):
    def __init__(self, number: int, chars: int) -> None:
        super().__init__()
        self.number = number
        self.chars = chars


class ChapterDone(Message):
    def __init__(
        self,
        number: int,
        title: str,
        words: int,
        seconds: float,
        prompt_tok: int,
        completion_tok: int,
        reasoning_tok: int,
        total_tok: int,
        path: str,
    ) -> None:
        super().__init__()
        self.number = number
        self.title = title
        self.words = words
        self.seconds = seconds
        self.prompt_tok = prompt_tok
        self.completion_tok = completion_tok
        self.reasoning_tok = reasoning_tok
        self.total_tok = total_tok
        self.path = path


class ChapterFailed(Message):
    def __init__(self, number: int, title: str, reason: str) -> None:
        super().__init__()
        self.number = number
        self.title = title
        self.reason = reason


class FilesChanged(Message):
    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self.paths = paths


class RunFinished(Message):
    def __init__(self, ok: bool, message: str) -> None:
        super().__init__()
        self.ok = ok
        self.message = message


class FileItem(ListItem):
    """One row in the output file list, carrying its absolute path."""

    def __init__(self, path: str, label: str) -> None:
        super().__init__()
        self.path = path
        self._label = label

    def compose(self):
        yield Label(self._label)


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


class BookBuilderApp(App):
    TITLE = "BookBuilder"
    SUB_TITLE = "live chapter generator"

    CSS = """
    Screen { layout: vertical; }
    #topbar { height: auto; padding: 1 1; }
    #config { height: auto; padding: 0 1; color: $text-muted; }
    #status { height: auto; padding: 0 1 1 1; color: $accent; text-style: bold; }
    #body { height: 1fr; }
    #left { width: 1fr; border-right: solid $accent; }
    #log { height: 1fr; }
    #right { width: 2fr; }
    #files { height: auto; max-height: 40%; border-bottom: solid $accent; }
    #viewer { height: 1fr; padding: 0 1; overflow-y: auto; }
    Select { width: 46; }
    Button { margin-left: 1; }
    """

    BINDINGS = [
        ("g", "generate", "Generate"),
        ("s", "stop", "Stop"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.selected_toc: str | None = None
        self.selected_style: str | None = None
        self.current_out_dir: str = ""
        self._current_title: str = ""
        self._stop = threading.Event()
        self.gen_worker = None

    # -- composition --------------------------------------------------------

    @staticmethod
    def _style_options() -> list[tuple[str, str]]:
        options = [("TOC default (no override)", "")]
        for key in sorted(PRESETS):
            options.append((f"{PRESETS[key].author} ({key})", key))
        return options

    def compose(self) -> ComposeResult:
        options = [(self._label_for(path), path) for path in discover_tocs(gb.HERE)]
        yield Header(show_clock=True)
        with Horizontal(id="topbar"):
            yield Select(
                options,
                prompt="Choose a TOC…",
                allow_blank=True,
                id="toc-select",
            )
            yield Select(
                self._style_options(),
                value="",
                allow_blank=False,
                id="style-select",
            )
            yield Button("Generate", id="generate", variant="primary")
            yield Button("Stop", id="stop", variant="error", disabled=True)
        yield Static("Select a TOC to begin.", id="config")
        yield Static("Idle", id="status")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield RichLog(id="log", wrap=True, highlight=True, markup=False)
            with Vertical(id="right"):
                yield ListView(id="files")
                yield Markdown(
                    "# Output\n\nPick a TOC and press **Generate**. Finished "
                    "chapters will show up here, rendered.",
                    id="viewer",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#log", RichLog).write(
            "Ready. Choose a TOC above, then press Generate (or hit 'g')."
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _label_for(path: str) -> str:
        try:
            return os.path.relpath(path, gb.HERE).replace("\\", "/")
        except ValueError:
            return os.path.basename(path)

    def output_dir_for(self, toc_path: str) -> str:
        stem = os.path.splitext(os.path.basename(toc_path))[0]
        return os.path.join(gb.HERE, "output", stem)

    def list_output_files(self, out_dir: str) -> list[str]:
        files: list[str] = []
        if not out_dir or not os.path.isdir(out_dir):
            return files
        main = os.path.join(out_dir, "final_book.md")
        if os.path.exists(main):
            files.append(main)
        chapters_dir = os.path.join(out_dir, "chapters")
        if os.path.isdir(chapters_dir):
            for name in sorted(os.listdir(chapters_dir)):
                if name.endswith(".md"):
                    files.append(os.path.join(chapters_dir, name))
        return files

    def set_files(self, paths: list[str]) -> None:
        try:
            view = self.query_one("#files", ListView)
        except Exception:
            return
        view.clear()
        for path in paths:
            label = os.path.relpath(path, self.current_out_dir).replace("\\", "/")
            view.append(FileItem(path, label))

    def show_file(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            text = f"Could not read `{path}`:\n\n    {exc}"
        self.query_one("#viewer", Markdown).update(text)

    def set_running(self, running: bool) -> None:
        self.query_one("#generate", Button).disabled = running
        self.query_one("#stop", Button).disabled = not running

    # -- event handlers -----------------------------------------------------

    @on(Select.Changed, "#toc-select")
    def on_toc_changed(self, event) -> None:
        value = event.value
        # Only a real, non-empty string path counts as a chosen TOC. This also
        # filters out the mount-time sentinels (Select.BLANK / Select.NULL),
        # which are not strings and must not trigger any work.
        if not isinstance(value, str) or not value:
            self.selected_toc = None
            return
        self.selected_toc = value
        self.current_out_dir = self.output_dir_for(value)
        stem = os.path.basename(self.current_out_dir)
        try:
            self.query_one("#config", Static).update(
                f"Output -> output/{stem}/   |   {gb.DEFAULT_MODEL} @ {gb.DEFAULT_BASE_URL}"
            )
        except Exception:
            pass
        self.set_files(self.list_output_files(self.current_out_dir))

    @on(Select.Changed, "#style-select")
    def on_style_changed(self, event) -> None:
        value = event.value
        self.selected_style = value if isinstance(value, str) and value else None

    @on(Button.Pressed, "#generate")
    def on_generate_pressed(self, _event) -> None:
        self.action_generate()

    @on(Button.Pressed, "#stop")
    def on_stop_pressed(self, _event) -> None:
        self.action_stop()

    @on(ListView.Selected, "#files")
    def on_file_selected(self, event) -> None:
        item = event.item
        if isinstance(item, FileItem):
            self.show_file(item.path)

    # -- actions ------------------------------------------------------------

    def action_generate(self) -> None:
        if not self.selected_toc:
            self.query_one("#log", RichLog).write("Choose a TOC first.")
            return
        if self.gen_worker and self.gen_worker.is_running:
            return
        self._stop.clear()
        self.query_one("#log", RichLog).clear()
        self.set_running(True)
        self.gen_worker = self.generate()

    def action_stop(self) -> None:
        self._stop.set()
        if self.gen_worker:
            self.gen_worker.cancel()
        self.query_one("#status", Static).update("Stopping after the current chapter…")

    # -- message handlers (called on the UI thread) --------------------------

    def on_log_msg(self, msg: LogMsg) -> None:
        self.query_one("#log", RichLog).write(msg.text)

    def on_status_msg(self, msg: StatusMsg) -> None:
        self.query_one("#status", Static).update(msg.text)

    def on_chapter_started(self, msg: ChapterStarted) -> None:
        self._current_title = msg.title
        self.query_one("#status", Static).update(
            f"Writing Chapter {msg.number}/{msg.total}: {msg.title}"
        )

    def on_chapter_progress(self, msg: ChapterProgress) -> None:
        self.query_one("#status", Static).update(
            f"Writing Chapter {msg.number}: {self._current_title}"
            f"  ·  {msg.chars:,} chars"
        )

    def on_chapter_done(self, msg: ChapterDone) -> None:
        line = (
            f"Ch {msg.number:>2} '{msg.title}' — {msg.words:,} words  |  "
            f"in {msg.prompt_tok:,} · out {msg.completion_tok:,} · "
            f"reason {msg.reasoning_tok:,} · total {msg.total_tok:,} tok  |  "
            f"{msg.seconds:.1f}s"
        )
        self.query_one("#log", RichLog).write(line)
        self.set_files(self.list_output_files(self.current_out_dir))
        self.show_file(msg.path)

    def on_chapter_failed(self, msg: ChapterFailed) -> None:
        self.query_one("#log", RichLog).write(
            f"Ch {msg.number:>2} '{msg.title}' FAILED — {msg.reason}"
        )

    def on_files_changed(self, msg: FilesChanged) -> None:
        self.set_files(msg.paths)

    def on_run_finished(self, msg: RunFinished) -> None:
        self.set_running(False)
        self.query_one("#status", Static).update(msg.message)
        self.query_one("#log", RichLog).write(msg.message)

    # -- the generation worker (runs in a background thread) -----------------

    @work(thread=True, group="generate", exclusive=True)
    def generate(self) -> None:
        toc_path = self.selected_toc
        out_dir = self.current_out_dir
        os.makedirs(out_dir, exist_ok=True)

        try:
            root = gb.load_toc(toc_path)
        except SystemExit as exc:
            self.post_message(RunFinished(False, f"Failed to load TOC: {exc}"))
            return

        chapters = root["chapters"]
        title = (
            root.get("title")
            or (root.get("metadata_defaults") or {}).get("title")
            or "Untitled Book"
        )
        builder_fn, _origin = gb.load_prompt_builder()

        settings = gb.Settings(
            toc_path=toc_path,
            out_dir=out_dir,
            base_url=gb.DEFAULT_BASE_URL,
            model=gb.DEFAULT_MODEL,
            credential=gb.DEFAULT_CREDENTIAL,
            temperature=0.7,
            max_tokens=None,
            overwrite=False,
            only=None,
            dry_run=False,
            verbose=False,
        )

        stem = os.path.basename(out_dir)
        self.post_message(
            StatusMsg(f"{title} · {len(chapters)} chapters · output/{stem}/")
        )
        self.post_message(
            LogMsg(f"Book: {title}  ·  {len(chapters)} chapters  ·  model {settings.model}")
        )
        self.post_message(FilesChanged(self.list_output_files(out_dir)))

        written = 0
        total = len(chapters)

        for index, chapter in enumerate(chapters):
            if self._stop.is_set() or (self.gen_worker and self.gen_worker.is_cancelled):
                self.post_message(LogMsg("Stopped by user."))
                break

            number, ch_title, prompt = gb.render_chapter(
                chapter, root, index, builder_fn, style=self.selected_style
            )
            target = os.path.join(out_dir, "chapters", f"chapter_{number:03d}.md")

            if (
                os.path.exists(target)
                and os.path.getsize(target) > 500
            ):
                self.post_message(
                    LogMsg(f"Ch {number:>2} '{ch_title}' already on disk — skipping")
                )
                self.post_message(FilesChanged(self.list_output_files(out_dir)))
                continue

            self.post_message(ChapterStarted(number, ch_title, total))
            budget = gb.token_budget(chapter, None)
            started = time.time()

            last = [0]

            def on_progress(chars: int, _n: int = number, _last=last) -> None:
                if chars - _last[0] >= 200:
                    _last[0] = chars
                    self.post_message(ChapterProgress(_n, chars))

            try:
                completion = gb.generate_chapter_text(
                    settings, prompt, budget, on_progress=on_progress
                )
            except Exception as exc:  # noqa: BLE001 - surface any engine error
                self.post_message(ChapterFailed(number, ch_title, str(exc)))
                continue

            if self._stop.is_set() or (self.gen_worker and self.gen_worker.is_cancelled):
                self.post_message(LogMsg("Stopped by user."))
                break

            seconds = time.time() - started
            prose = gb.clean_completion(completion.text, ch_title)
            if not prose:
                self.post_message(ChapterFailed(number, ch_title, "no usable text"))
                continue

            gb.write_chapter(out_dir, number, ch_title, prose)
            gb.recompile_book(out_dir, title)
            written += 1

            usage = completion.usage
            self.post_message(
                ChapterDone(
                    number=number,
                    title=ch_title,
                    words=len(prose.split()),
                    seconds=seconds,
                    prompt_tok=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    completion_tok=getattr(usage, "completion_tokens", 0) if usage else 0,
                    reasoning_tok=getattr(usage, "reasoning_tokens", 0) if usage else 0,
                    total_tok=getattr(usage, "total_tokens", 0) if usage else 0,
                    path=target,
                )
            )
            self.post_message(FilesChanged(self.list_output_files(out_dir)))

        self.post_message(
            RunFinished(True, f"Done — {written} chapter(s) written to output/{stem}/")
        )


def main() -> None:
    BookBuilderApp().run()


if __name__ == "__main__":
    main()
