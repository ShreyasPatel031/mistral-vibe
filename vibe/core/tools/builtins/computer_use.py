from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import importlib.util
import os
from typing import TYPE_CHECKING, Any, final
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from vibe.core.config import DEFAULT_MISTRAL_API_ENV_KEY, VibeConfigSchema
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import (
    PermissionContext,
    PermissionScope,
    RequiredPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolStreamEvent
from vibe.utils.api_keys import resolve_api_key
from vibe.utils.tool_presentation import ToolEffectKind

if TYPE_CHECKING:
    from vibe.core.types import ToolCallEvent, ToolResultEvent

_MISTRAL_API_BASE = "https://api.mistral.ai/v1"
_STEP_CAP = 40
_WALL_TIMEOUT_SECONDS = 120
_HEARTBEAT_SECONDS = 3.0
_QUEUE_POLL_SECONDS = 0.25


class ComputerUseStep(BaseModel):
    step: int
    action: str
    thought: str = ""
    url: str = ""


class ComputerUseArgs(BaseModel):
    url: str = Field(description="Public URL to open before starting the task.")
    task: str = Field(
        min_length=1, description="What to accomplish on the site, in concrete terms."
    )
    intent: str = Field(
        default="",
        description="Who the user is and what they are trying to achieve, so the agent can behave like them.",
    )
    max_steps: int | None = Field(
        default=None, description="Override the configured step budget for this run."
    )
    show_browser: bool = Field(
        default=True,
        description="Open a visible Chromium window so you can watch the agent work.",
    )


class ComputerUseResult(BaseModel):
    url: str
    task: str
    completed: bool
    final_url: str
    summary: str
    steps: list[ComputerUseStep] = Field(default_factory=list)
    num_steps: int = 0
    budget_exhausted: bool = False


class ComputerUseConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK

    model: str = Field(
        default="mistral-small-latest",
        description="Mistral model driving the browser. Needs vision for screenshots.",
    )
    api_base: str = Field(
        default=_MISTRAL_API_BASE, description="OpenAI-compatible Mistral endpoint."
    )
    max_steps: int = Field(
        default=12, description="Default step budget for a single task."
    )
    headless: bool = Field(
        default=False,
        description="Hide the browser window. Prefer show_browser=false on each call instead.",
    )
    viewport_width: int = Field(default=1280)
    viewport_height: int = Field(default=800)
    wall_timeout_seconds: int = Field(
        default=_WALL_TIMEOUT_SECONDS,
        description="Hard cap on wall time for one browser task.",
    )


def _is_meaningful(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str | list | dict | tuple) and not value:
        return False
    return True


def _action_label(action: Any) -> str:
    if action is None:
        return "—"
    if isinstance(action, list):
        return "; ".join(_action_label(item) for item in action)

    payload = getattr(action, "root", None) or action
    dumped = (
        payload.model_dump(exclude_none=True)
        if hasattr(payload, "model_dump")
        else None
    )
    if not isinstance(dumped, dict) or not dumped:
        return str(action)[:200]

    name, arguments = next(iter(dumped.items()))
    if not isinstance(arguments, dict):
        return f"{name}: {arguments}"[:200]

    detail = ", ".join(
        f"{key}={value}" for key, value in arguments.items() if _is_meaningful(value)
    )
    return (f"{name} — {detail}" if detail else str(name))[:200]


def _thought(model_output: Any) -> str:
    for field_name in ("next_goal", "evaluation_previous_goal", "thinking"):
        value = getattr(model_output, field_name, None)
        if value:
            return str(value).strip()[:300]
    return ""


def _step_from_history_item(item: Any, step_no: int) -> ComputerUseStep:
    model_output = getattr(item, "model_output", None)
    state = getattr(item, "state", None)
    return ComputerUseStep(
        step=step_no,
        action=_action_label(getattr(model_output, "action", None)),
        thought=_thought(model_output),
        url=str(getattr(state, "url", "") or ""),
    )


def _format_step_message(step: ComputerUseStep, max_steps: int) -> str:
    lines = [f"Step {step.step}/{max_steps}: {step.action}"]
    if step.url:
        lines.append(f"  @ {step.url}")
    if step.thought:
        lines.append(f"  → {step.thought}")
    return "\n".join(lines)


class ComputerUse(
    BaseTool[ComputerUseArgs, ComputerUseResult, ComputerUseConfig, BaseToolState],
    ToolUIData[ComputerUseArgs, ComputerUseResult],
):
    effect_kind = ToolEffectKind.TOOL

    @classmethod
    def is_available(cls, config: VibeConfigSchema | None = None) -> bool:
        if importlib.util.find_spec("browser_use") is None:
            return False
        return bool(resolve_api_key(DEFAULT_MISTRAL_API_ENV_KEY))

    @staticmethod
    def _normalize_url(url: str) -> str:
        raw = url.lstrip("/") if url.startswith("//") else url
        return raw if raw.startswith(("http://", "https://")) else "https://" + raw

    def resolve_permission(self, args: ComputerUseArgs) -> PermissionContext | None:
        if self.config.permission in {ToolPermission.ALWAYS, ToolPermission.NEVER}:
            return PermissionContext(permission=self.config.permission)

        parsed = urlparse(self._normalize_url(args.url))
        domain = parsed.netloc or parsed.path.split("/")[0]
        if not domain:
            return None

        return PermissionContext(
            permission=ToolPermission.ASK,
            required_permissions=[
                RequiredPermission(
                    scope=PermissionScope.URL_PATTERN,
                    invocation_pattern=domain,
                    session_pattern=domain,
                    label=f"driving a browser on {domain}",
                )
            ],
        )

    def _stream_event(self, message: str, ctx: InvokeContext | None) -> ToolStreamEvent:
        return ToolStreamEvent(
            tool_name=self.get_name(),
            message=message if message.endswith("\n") else f"{message}\n",
            tool_call_id=ctx.tool_call_id if ctx else "",
        )

    def _resolve_headless(self, args: ComputerUseArgs) -> bool:
        if os.environ.get("COMPUTER_USE_HEADLESS", "").lower() in {"1", "true", "yes"}:
            return True
        if os.environ.get("COMPUTER_USE_HEADED", "").lower() in {"1", "true", "yes"}:
            return False
        return not args.show_browser

    def _make_browser_profile(self, headless: bool, profile_factory: Any) -> Any:
        return profile_factory(
            headless=headless,
            demo_mode=not headless,
            disable_security=True,
            enable_default_extensions=False,
            captcha_solver=False,
            highlight_elements=False,
            keep_alive=False,
            minimum_wait_page_load_time=0.5,
            wait_for_network_idle_page_load_time=0.5,
            wait_between_actions=0.25,
            window_size={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
        )

    @staticmethod
    def _load_browser_modules() -> tuple[Any, Any, Any]:
        browser_use = importlib.import_module("browser_use")
        profile_module = importlib.import_module("browser_use.browser.profile")
        return browser_use.Agent, browser_use.ChatOpenAI, profile_module.BrowserProfile

    def _create_agent(
        self,
        url: str,
        args: ComputerUseArgs,
        api_key: str,
        headless: bool,
        agent_factory: Any,
        chat_factory: Any,
        profile_factory: Any,
    ) -> Any:
        return agent_factory(
            task=self._build_prompt(url, args),
            llm=chat_factory(
                model=self.config.model,
                api_key=api_key,
                base_url=self.config.api_base,
                temperature=0,
                max_retries=4,
                timeout=90.0,
            ),
            browser_profile=self._make_browser_profile(headless, profile_factory),
            use_vision=True,
            use_judge=False,
            max_actions_per_step=2,
            calculate_cost=True,
            extend_system_message=(
                "Complete the task in as few steps as possible. "
                "Call done as soon as every constraint is satisfied on the page."
            ),
        )

    @final
    async def run(
        self, args: ComputerUseArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ComputerUseResult, None]:
        if importlib.util.find_spec("browser_use") is None:
            raise ToolError(
                "browser-use is not installed. Install it with "
                "`uv pip install browser-use` then `playwright install chromium`."
            )

        agent_factory, chat_factory, profile_factory = self._load_browser_modules()

        api_key = resolve_api_key(DEFAULT_MISTRAL_API_ENV_KEY)
        if not api_key:
            raise ToolError(
                f"{DEFAULT_MISTRAL_API_ENV_KEY} environment variable not set."
            )

        url = self._normalize_url(args.url)
        self._validate_url(url)
        max_steps = min(args.max_steps or self.config.max_steps, _STEP_CAP)
        wall_timeout = self.config.wall_timeout_seconds

        os.environ.setdefault("BROWSER_USE_STEP_TIMEOUT", "45")
        os.environ.setdefault("BROWSER_USE_ACTION_TIMEOUT_S", "60")

        headless = self._resolve_headless(args)
        open_msg = (
            f"Opening visible browser → {url} …"
            if not headless
            else f"Opening {url} (headless) …"
        )
        yield self._stream_event(open_msg, ctx)
        yield self._stream_event(
            "Launching Chromium (extensions disabled for speed)…", ctx
        )

        agent = self._create_agent(
            url, args, api_key, headless, agent_factory, chat_factory, profile_factory
        )

        progress: asyncio.Queue[str] = asyncio.Queue()
        completed_steps = 0

        async def on_step_end(running_agent: Any) -> None:
            nonlocal completed_steps
            history = getattr(running_agent, "history", None)
            items = list(getattr(history, "history", None) or [])
            if not items:
                return
            completed_steps = len(items)
            step = _step_from_history_item(items[-1], completed_steps)
            await progress.put(_format_step_message(step, max_steps))

        run_task = asyncio.create_task(
            agent.run(max_steps=max_steps, on_step_end=on_step_end)
        )

        try:
            async for message in self._drain_with_heartbeat(
                progress,
                run_task,
                max_steps,
                completed_steps_ref=lambda: completed_steps,
            ):
                yield self._stream_event(message, ctx)

            history = await asyncio.wait_for(run_task, timeout=wall_timeout)
        except TimeoutError:
            run_task.cancel()
            raise ToolError(
                f"Browser task timed out after {wall_timeout}s. "
                "Try a simpler task or raise max_steps."
            ) from None
        except Exception as exc:
            run_task.cancel()
            raise ToolError(f"Browser agent failed: {exc}") from exc

        result = self._build_result(history, url, args, max_steps)
        yield self._stream_event(self._format_completion_message(result), ctx)
        yield result

    @staticmethod
    async def _drain_with_heartbeat(
        progress: asyncio.Queue[str],
        run_task: asyncio.Task[Any],
        max_steps: int,
        *,
        completed_steps_ref: Any,
    ) -> AsyncGenerator[str, None]:
        elapsed = 0.0
        while not run_task.done():
            try:
                yield await asyncio.wait_for(progress.get(), _HEARTBEAT_SECONDS)
            except TimeoutError:
                elapsed += _HEARTBEAT_SECONDS
                done = completed_steps_ref()
                phase = (
                    "launching browser"
                    if done == 0 and elapsed < 45
                    else "waiting on browser/model"
                )
                yield (
                    f"… still working ({int(elapsed)}s, "
                    f"step {done + 1}/{max_steps}, {phase})"
                )
        while not progress.empty():
            yield progress.get_nowait()

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolError(
                f"Invalid URL scheme: {parsed.scheme}. Must be http or https."
            )
        if not parsed.netloc:
            raise ToolError("URL must include a host.")

    @staticmethod
    def _build_prompt(url: str, args: ComputerUseArgs) -> str:
        lines = [f"Open {url} and complete the task below."]
        if args.intent:
            lines.append(f"You are acting for this user: {args.intent}")
        lines.append(f"Task: {args.task}")
        lines.append(
            "Use the site's own controls — filters, dropdowns, and date pickers — "
            "rather than typing every constraint into a search box."
        )
        lines.append(
            "Stop as soon as every part of the task is satisfied on the page. "
            "Call done immediately when finished."
        )
        return "\n".join(lines)

    @staticmethod
    def _build_result(
        history: Any, url: str, args: ComputerUseArgs, max_steps: int
    ) -> ComputerUseResult:
        items = list(getattr(history, "history", None) or [])
        steps = [_step_from_history_item(item, i) for i, item in enumerate(items, 1)]

        final_url = next((step.url for step in reversed(steps) if step.url), url)
        completed = bool(_call_or_default(history, "is_done", False))
        summary = str(_call_or_default(history, "final_result", "") or "")

        return ComputerUseResult(
            url=url,
            task=args.task,
            completed=completed,
            final_url=final_url,
            summary=summary,
            steps=steps,
            num_steps=len(steps),
            budget_exhausted=len(steps) >= max_steps and not completed,
        )

    @staticmethod
    def _format_completion_message(result: ComputerUseResult) -> str:
        status = "completed" if result.completed else "stopped"
        lines = [
            f"Done ({status}) — {result.num_steps} steps",
            f"Final page: {result.final_url}",
        ]
        if result.summary:
            lines.append(result.summary.strip())
        return "\n".join(lines)

    def get_result_extra(self, result: ComputerUseResult) -> str | None:
        lines = [
            "Browser trace:",
            *[
                f"  {s.step}. {s.action}" + (f" @ {s.url}" if s.url else "")
                for s in result.steps
            ],
        ]
        if result.summary:
            lines.append(f"Summary: {result.summary.strip()}")
        lines.append(f"Final: {result.final_url}")
        return "\n".join(lines)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ComputerUseArgs):
            return ToolCallDisplay(
                summary="computer_use",
                verb="Running",
                message="computer_use",
                settled_verb="Ran",
                settled_message="computer_use",
            )

        parsed = urlparse(cls._normalize_url(event.args.url))
        message = f"{parsed.netloc or event.args.url[:50]} — {event.args.task[:80]}"
        return ToolCallDisplay(
            summary=f"Browsing: {message}",
            verb="Browsing",
            message=message,
            settled_verb="Browsed",
            settled_message=message,
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ComputerUseResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        result = event.result
        snippet = result.summary.strip().split("\n")[0][:80] if result.summary else ""
        message = result.final_url
        if snippet:
            message = f"{result.final_url} — {snippet}"
        if result.budget_exhausted:
            return ToolResultDisplay(
                success=False,
                verb="Ran out of steps",
                message=message,
                suffix=f"({result.num_steps} steps)",
            )
        return ToolResultDisplay(
            success=result.completed,
            verb="Completed" if result.completed else "Stopped",
            message=message,
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Launching Chromium…"


def _call_or_default(target: Any, method_name: str, default: Any) -> Any:
    method = getattr(target, method_name, None)
    if not callable(method):
        return default
    try:
        return method()
    except Exception:
        return default
