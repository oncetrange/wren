"""The full-screen session (`wren --fullscreen`): wren draws the whole terminal.

    ┌──────────────────────────────────────────┐
    │ the conversation (scrolls: wheel, PgUp)   │  transcript.py
    │ ⠋ Running bash…                          │  status: spinner, subagents, queued messages
    │ ──────────────────────────────────────── │
    │ › the input line, always there            │  or a question: a picker, "Allow?"
    │ ⏵ ask before edits · shift+tab · kimi     │  toolbar
    └──────────────────────────────────────────┘

The session itself (the agent, slash commands) runs on the main thread as in
the inline REPL, and the UI on a thread of its own. Lines typed while the
session is idle go to it through a queue; typed while it works, they go to the
agent as steering. Ctrl-C while it works sends the main thread SIGINT, so
everything that already handles KeyboardInterrupt (tools, retries, /loop)
behaves as it does inline. A question (a picker, approving a tool call) blocks
the main thread until the UI answers it.

Before the UI starts and after it stops, output and questions fall back to the
inline forms, so startup prompts and the end-of-session memory review work as
they always did.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples, to_formatted_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    Dimension,
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.processors import AppendAutoSuggestion
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.text import Text

from wren import mentions
from wren.agent.loop import Agent
from wren.agent.predict import Predictor
from wren.cli import commands, render
from wren.cli.app import run_prompt
from wren.cli.completion import SlashMenu
from wren.cli.keys import newline_bindings
from wren.cli.repl import TOOLBAR_STYLE, Repl
from wren.cli.screen import Screen
from wren.cli.terminal import distinguish_shift_enter, register_shift_enter
from wren.cli.transcript import Block, Transcript, TranscriptControl, to_lines
from wren.cli.ui import RichUI
from wren.config import CONFIG_DIR, Config
from wren.llm.types import Message
from wren.memory import Memories
from wren.settings import Settings
from wren.tools import Tool, ToolContext

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
MAX_PICKER_ROWS = 10


@dataclass
class Question:
    """Something the session waits for the user to answer: one of `options`,
    or (without options) a line of text."""

    message: str
    options: list[tuple[Any, str]] | None = None
    index: int = 0
    future: Future = field(default_factory=Future)


class View:
    """What the UI thread draws, set from either thread."""

    def __init__(self, color_system: str | None):
        self.transcript = Transcript(color_system)
        self.control = TranscriptControl(self.transcript)
        self.status: RenderableType | None = None  # the spinner line, while something runs
        self.question: Question | None = None
        self.app: Application | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.context: contextvars.Context | None = None  # the UI thread's, with its app
        self.running = False
        self.transcript.changed = self.redraw
        # Set by FullscreenRepl: where an answered question hands the keyboard back.
        self.on_question: Callable[[], None] = lambda: None

    def redraw(self) -> None:
        if self.app is not None and self.running:
            self.app.invalidate()

    def call(self, fn: Callable[[], Any]) -> None:
        """Run `fn` on the UI thread."""
        if self.loop is not None and self.running:
            self.loop.call_soon_threadsafe(fn, context=self.context)

    def set_status(self, message: RenderableType | None) -> None:
        self.status = message
        self.redraw()

    # --- questions ---------------------------------------------------------------

    def ask(self, question: Question) -> Any:
        """From the session's thread: show the question and wait for the answer."""
        def show() -> None:
            self.question = question
            self.on_question()
        self.call(show)
        return question.future.result()

    def answer(self, value: Any = None, error: BaseException | None = None) -> None:
        """On the UI thread: the current question is answered (or cancelled with `error`)."""
        question, self.question = self.question, None
        self.on_question()
        if question is None:
            return
        if error is not None:
            question.future.set_exception(error)
        else:
            question.future.set_result(value)


class FullscreenScreen(Screen):
    def __init__(self, console: Console, view: View):
        super().__init__(console, interactive=True)
        self.view = view

    @property
    def width(self) -> int:
        app = self.view.app
        return app.output.get_size().columns if app is not None and self.view.running else self.console.width

    def print(self, *objects: Any, **kwargs: Any) -> None:
        if not self.view.running:
            self.console.print(*objects, **kwargs)
        self.view.transcript.add(*objects)

    def clear(self) -> None:
        self.view.transcript.clear()
        self.view.control.to_bottom()

    def pick[T](self, message: str, options: Sequence[tuple[T, str]], default: T | None = None) -> T | None:
        if not self.view.running:
            return super().pick(message, options, default=default)
        if not options:
            return None
        values = [value for value, _ in options]
        index = values.index(default) if default in values else 0
        return self.view.ask(Question(message, list(options), index))

    def ask_text(self, message: str = "  › ") -> str:
        if not self.view.running:
            return super().ask_text(message)
        return self.view.ask(Question(message))

    def status(self, message: str) -> contextlib.AbstractContextManager[Any]:
        if not self.view.running:
            return super().status(message)
        return _Status(self.view, message)

    def page(self, *renderables: RenderableType) -> None:
        self.print(*renderables)  # the transcript scrolls

    def run_program(self, argv: list[str]) -> int:
        if not self.view.running or self.view.loop is None:
            return super().run_program(argv)

        async def run() -> int:
            return await run_in_terminal(lambda: subprocess.run(argv, check=False).returncode)

        done: Future = Future()

        def start() -> None:
            task = asyncio.ensure_future(run())
            task.add_done_callback(lambda t: done.set_exception(e) if (e := t.exception()) else
                                   done.set_result(t.result()))
        self.view.call(start)
        return done.result()


class _Status:
    def __init__(self, view: View, message: str):
        self.view, self.message = view, message

    def __enter__(self) -> _Status:
        self.view.set_status(Text(self.message))
        return self

    def update(self, message: str) -> None:
        self.view.set_status(Text(message))

    def __exit__(self, *exc: object) -> None:
        self.view.set_status(None)


class FullscreenUI(RichUI):
    """The agent's events, as transcript blocks and the status line."""

    def __init__(self, console: Console, background: str = "dark"):
        super().__init__(console, interactive=True, background=background)
        self.view = View(console.color_system)
        self.screen = FullscreenScreen(console, self.view)
        self._streamed = ""
        self._block: Block | None = None

    def _print(self, *renderables: Any, **kwargs: Any) -> None:
        self.screen.print(*renderables, **kwargs)

    def tool_started(self, name: str, label: str) -> None:
        super().tool_started(name, label)
        if not self._depth:
            self._spin(Text(f"Running {name}…", style="dim"))

    def render_history(self, messages: list[Message], tools: dict[str, Tool], ctx: ToolContext) -> None:
        self.view.transcript.add(*render.history(messages, tools, ctx, self.code_theme, self.screen.width))

    def _stream(self, text: str, kind: str, style: str) -> None:
        self._stop_spin()
        if self._stream_kind != kind:
            self._end_stream()
            self._stream_kind, self._block = kind, self.view.transcript.add()
        self._streamed += text
        assert self._block is not None
        shown = (Markdown(self._streamed, code_theme=self.code_theme) if kind == "text"
                 else Text(self._streamed, style=style))
        self.view.transcript.update(self._block, shown)

    def _end_stream(self) -> None:
        self._stream_kind, self._block, self._streamed = None, None, ""

    def _spin(self, message: str | Text) -> None:
        self.view.set_status(message)

    def _stop_spin(self) -> None:
        self.view.set_status(None)


class FullscreenRepl(Repl):
    def __init__(self, agent: Agent, ui: FullscreenUI, config: Config, settings: Settings,
                 memories: Memories | None = None):
        super().__init__(agent, ui, config, settings, memories)
        self.view = ui.view
        self.inbox: queue.Queue[str | None] = queue.Queue()  # None: end the session
        self.busy = False  # the session is handling a line; Ctrl-C interrupts it
        self._busy_lock = threading.Lock()
        self._session_thread = threading.get_ident()
        self._usage = ""
        self._hint = ""  # shown in the toolbar until the next key
        self._error: BaseException | None = None

    # --- the session's thread --------------------------------------------------

    def loop(self) -> int:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._session_thread = threading.get_ident()
        self.menu = SlashMenu(self.completions, mentions.FileIndex(self.agent.ctx.cwd))
        self.predictor = Predictor(self.agent, on_ready=self._prediction_ready)
        self.app = self.build()
        started = threading.Event()
        ui = threading.Thread(target=contextvars.copy_context().run, args=(self._run_app, started),
                              name="wren-ui", daemon=True)
        ui.start()
        started.wait()
        self.banner()
        try:
            while True:
                try:
                    if self._step() == "exit":
                        break
                except KeyboardInterrupt:  # Ctrl-C landed between two steps
                    continue
        finally:
            self.view.call(self.app.exit)
            ui.join()
        if self._error is not None:
            raise self._error
        return 0

    def _step(self) -> str | None:
        text = self.inbox.get()
        try:
            if text is None:
                self.save_memories()
                return "exit"
            return self.handle(text)
        finally:
            with self._busy_lock:
                self.busy = False
            self.view.call(self._idle)

    def run(self, text: str) -> None:
        run_prompt(self.agent, text, commands.BUILTIN_NAMES)
        if left := self.agent.unsent():
            self.prefill = "\n".join(left)
            self.screen.print("[dim]not sent to wren; it's back in the input line[/]")

    def usage_line(self) -> None:
        u = render.usage(self.agent.estimated_context(), self.agent.model.context_window,
                         self.agent.usage.output_tokens, self.agent.cost)
        self._usage = u.plain.strip()
        self.view.redraw()

    def _prediction_ready(self) -> None:
        self.view.call(self._show_prediction)

    # --- the UI thread ---------------------------------------------------------

    def _run_app(self, started: threading.Event) -> None:
        def pre_run() -> None:
            self.view.loop = asyncio.get_running_loop()
            self.view.context = contextvars.copy_context()
            self.view.running = True
            self.app.create_background_task(self._tick())
            started.set()
        try:
            register_shift_enter()
            with distinguish_shift_enter():
                self.app.run(pre_run=pre_run, handle_sigint=False)
        except BaseException as e:  # the session must not wait for a UI that is gone
            self._error = e
        finally:
            self.view.running = False
            started.set()
            if (question := self.view.question) is not None and not question.future.done():
                question.future.set_exception(KeyboardInterrupt())
            if self._error is not None:
                self.inbox.put(None)

    async def _tick(self) -> None:
        """Animate the spinner while something runs."""
        while True:
            await asyncio.sleep(0.1)
            if self.view.status is not None:
                self.app.invalidate()

    def build(self) -> Application:
        view = self.view
        # Focused while a picker is open, so what is typed meanwhile goes nowhere.
        self._picker = FormattedTextControl(self._picker_lines, focusable=True)
        self.input = Buffer(multiline=True, history=FileHistory(str(CONFIG_DIR / "history")),
                            accept_handler=self._accept, name="input")
        self.answer = Buffer(multiline=True, accept_handler=self._accept_answer, name="answer")
        self.menu.attach(self.input)
        self.input.on_text_changed += lambda _: setattr(self, "_hint", "")

        asking_text = Condition(lambda: view.question is not None and view.question.options is None)
        picking = Condition(lambda: view.question is not None and view.question.options is not None)
        no_question = Condition(lambda: view.question is None)

        def prefix(text: str) -> Callable[[int, int], StyleAndTextTuples]:
            return lambda line, wrap: [("class:prompt", text if line == 0 and wrap == 0 else "  ")]

        body = HSplit([
            Window(view.control, wrap_lines=False),
            ConditionalContainer(Window(FormattedTextControl(self._status_lines), dont_extend_height=True),
                                 filter=Condition(lambda: view.status is not None or bool(self.agent.queued))),
            Window(height=1, char="─", style="class:rule"),
            ConditionalContainer(
                Window(BufferControl(self.input, input_processors=[AppendAutoSuggestion()]),
                       height=Dimension(min=1, max=12), wrap_lines=True, dont_extend_height=True,
                       get_line_prefix=prefix("› ")),
                filter=no_question),
            ConditionalContainer(Window(self._picker, dont_extend_height=True), filter=picking),
            ConditionalContainer(
                HSplit([
                    Window(FormattedTextControl(lambda: self._question_header()), dont_extend_height=True),
                    Window(BufferControl(self.answer), height=Dimension(min=1, max=8), wrap_lines=True,
                           dont_extend_height=True, get_line_prefix=prefix("› ")),
                ]),
                filter=asking_text),
            Window(FormattedTextControl(self._toolbar_line), height=1, style="class:bottom-toolbar"),
        ])
        view.on_question = self._focus
        app: Application = Application(
            layout=Layout(body, focused_element=self.input),
            key_bindings=merge_key_bindings([self._bindings(picking), newline_bindings(),
                                             self.menu.bindings()]),
            style=TOOLBAR_STYLE, full_screen=True, mouse_support=True, min_redraw_interval=0.05,
        )
        view.app = app
        return app

    def _focus(self) -> None:
        """Give the keyboard to the question being asked, or back to the input line."""
        question = self.view.question
        if question is None:
            self.app.layout.focus(self.input)
        else:
            self.view.control.to_bottom()  # what is being asked about is at the end
            if question.options is None:
                self.answer.reset()
                self.app.layout.focus(self.answer)
            else:
                self.app.layout.focus(self._picker)
        self.app.invalidate()

    def _bindings(self, picking: Condition) -> KeyBindings:
        kb = KeyBindings()
        view = self.view

        @kb.add("enter", filter=~self.menu.is_open & ~picking)
        def _(event) -> None:
            event.current_buffer.validate_and_handle()

        @kb.add("c-c")
        def _(event) -> None:
            if view.question is not None:
                cancel = KeyboardInterrupt() if view.question.options is None else None
                view.answer(None, error=cancel)
            elif self.input.text:
                self.input.reset()
            elif not self._interrupt():
                self._hint = "Ctrl-D quits"

        @kb.add("c-d")
        def _(event) -> None:
            buf = event.current_buffer
            if buf.text:
                buf.delete()
            elif view.question is not None and view.question.options is None:
                view.answer(error=EOFError())
            elif view.question is None and not self.busy:
                self._send(None)

        @kb.add("s-tab")
        def _(event) -> None:
            self.agent.permissions.cycle()

        @kb.add("pageup")
        def _(event) -> None:
            view.control.page(1)

        @kb.add("pagedown")
        def _(event) -> None:
            view.control.page(-1)

        @kb.add("up", filter=picking)
        @kb.add("c-p", filter=picking)
        def _(event) -> None:
            self._move(-1)

        @kb.add("down", filter=picking)
        @kb.add("c-n", filter=picking)
        def _(event) -> None:
            self._move(1)

        @kb.add("enter", filter=picking)
        def _(event) -> None:
            question = view.question
            assert question is not None and question.options is not None
            view.answer(question.options[question.index][0])

        @kb.add("escape", filter=picking, eager=True)
        def _(event) -> None:
            view.answer(None)

        return kb

    def _move(self, step: int) -> None:
        question = self.view.question
        if question is not None and question.options:
            question.index = (question.index + step) % len(question.options)

    def _accept(self, buf: Buffer) -> bool:
        text = buf.text.strip()
        if text:
            self.view.control.to_bottom()
            if self.busy:
                self.agent.steer(text)
            else:
                self.view.transcript.add(render.prompt(text))
                self._send(text)
        return False  # clears the input

    def _accept_answer(self, buf: Buffer) -> bool:
        self.view.answer(buf.text)
        return False

    def _send(self, text: str | None) -> None:
        with self._busy_lock:
            self.busy = True
        self.inbox.put(text)

    def _interrupt(self) -> bool:
        """Ctrl-C while the session works: stop it, as Ctrl-C does inline."""
        with self._busy_lock:
            if self.busy:
                signal.pthread_kill(self._session_thread, signal.SIGINT)
                return True
        return False

    def _idle(self) -> None:
        """The session is done with a line: what it gave back goes in the input line."""
        prefill, self.prefill = self.prefill, ""
        if prefill:
            self.input.text = prefill + ("\n" + self.input.text if self.input.text else "")
            self.input.cursor_position = len(self.input.text)
        self._show_prediction()

    def _show_prediction(self) -> None:
        self.menu.show_prediction(self.predictor.take(), self.input)
        self.app.invalidate()

    # --- drawing -----------------------------------------------------------------

    def _width(self) -> int:
        return self.app.output.get_size().columns

    def _status_lines(self) -> StyleAndTextTuples:
        shown: list[RenderableType] = []
        if (status := self.view.status) is not None:
            frame = SPINNER[int(time.monotonic() * 10) % len(SPINNER)]
            line = Text(frame + " ", style="cyan")
            line.append_text(Text(status) if isinstance(status, str) else status)  # type: ignore[arg-type]
            shown.append(line)
        if n := self.agent.queued:
            shown.append(Text(f"⏎ {n} message{'s' if n != 1 else ''} for wren after this step", style="dim"))
        return _join(to_lines(shown, self._width(), self.view.transcript.color_system))

    def _picker_lines(self) -> StyleAndTextTuples:
        question = self.view.question
        if question is None or question.options is None:
            return []
        lines: StyleAndTextTuples = [("bold", question.message), ("class:menu.hint", "  ↑↓ ↵ · Esc cancels\n")]
        start = max(0, min(question.index - MAX_PICKER_ROWS // 2, len(question.options) - MAX_PICKER_ROWS))
        for i in range(start, min(len(question.options), start + MAX_PICKER_ROWS)):
            label = question.options[i][1]
            lines.append(("class:menu.selected", f" ❯ {label} ") if i == question.index else ("", f"   {label}"))
            lines.append(("", "\n"))
        if len(question.options) > MAX_PICKER_ROWS:
            lines.append(("class:menu.hint", f"   {question.index + 1}/{len(question.options)}"))
        return lines

    def _question_header(self) -> StyleAndTextTuples:
        question = self.view.question
        message = question.message.strip() if question is not None else ""
        lines: StyleAndTextTuples = [("bold", message), ("", "  ")] if message and message != "›" else []
        return [*lines, ("class:menu.hint", "Enter answers · Ctrl-C cancels")]

    def _toolbar_line(self) -> StyleAndTextTuples:
        parts = to_formatted_text(self._toolbar())
        extra = []
        if below := self.view.control.below:
            extra.append(f"↓ {below} more lines (PgDn)")
        if self._usage:
            extra.append(self._usage)
        if self._hint:
            extra.append(self._hint)
        return [*parts, *(("ansigray", " · " + e) for e in extra)]


def _join(lines: list[StyleAndTextTuples]) -> StyleAndTextTuples:
    out: StyleAndTextTuples = []
    for i, line in enumerate(lines):
        if i:
            out.append(("", "\n"))
        out += line
    return out
