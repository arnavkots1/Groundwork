from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.message import Message
from pathlib import Path

from env_loader import load_local_env
from rate_limiter import RateLimitError, acquire_slot, record_failure, record_success, retry_after_seconds


load_local_env()


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        headers: dict[str, str] | None = None,
        retryable: bool = False,
        model: str | None = None,
        provider: str | None = None,
        quota_exhausted: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.headers = headers or {}
        self.retryable = retryable
        self.model = model
        self.provider = provider
        self.quota_exhausted = quota_exhausted

    @property
    def is_rate_limit(self) -> bool:
        return self.status_code == 429

    @property
    def is_unavailable(self) -> bool:
        return self.status_code in {400, 404, 410, 422}


@dataclass(frozen=True)
class JsonResponse:
    body: dict
    headers: dict[str, str]


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    kind: str
    model: str
    env_key: str | None = None
    base_url: str | None = None
    env_base_url: str | None = None


PROVIDERS: dict[str, ProviderConfig] = {
    "OpenAI API": ProviderConfig(
        name="OpenAI API",
        kind="openai_compatible",
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        env_key="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        env_base_url="OPENAI_BASE_URL",
    ),
    "Gemini": ProviderConfig(
        name="Gemini",
        kind="gemini_interactions",
        model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        env_key="GEMINI_API_KEY",
    ),
    "NVIDIA NIM": ProviderConfig(
        name="NVIDIA NIM",
        kind="openai_compatible",
        model=os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
        env_key="NVIDIA_API_KEY",
        base_url="https://integrate.api.nvidia.com/v1",
        env_base_url="NVIDIA_BASE_URL",
    ),
    "Kimi / Moonshot": ProviderConfig(
        name="Kimi / Moonshot",
        kind="openai_compatible",
        model=os.environ.get("MOONSHOT_MODEL", "kimi-k2-0905-preview"),
        env_key="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.ai/v1",
        env_base_url="MOONSHOT_BASE_URL",
    ),
    "Anthropic API": ProviderConfig(
        name="Anthropic API",
        kind="anthropic",
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
        env_key="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1",
        env_base_url="ANTHROPIC_BASE_URL",
    ),
    # Subscription-backed lead routes. These shell out to a locally installed
    # agent CLI instead of spending API credits. env_key is None because auth
    # lives in the CLI's own login, not in an environment variable.
    "Claude Code CLI": ProviderConfig(
        name="Claude Code CLI",
        kind="agent_cli",
        model=os.environ.get("CLAUDE_CLI_MODEL", "claude-code-cli"),
        env_key=None,
    ),
    "Codex CLI": ProviderConfig(
        name="Codex CLI",
        kind="agent_cli",
        model=os.environ.get("CODEX_CLI_MODEL", "codex-cli"),
        env_key=None,
    ),
    "Cursor Agent CLI": ProviderConfig(
        name="Cursor Agent CLI",
        kind="agent_cli",
        model=os.environ.get("CURSOR_CLI_MODEL", "auto"),
        env_key=None,
    ),
}

# Default headless invocations. Override with CLAUDE_CLI_COMMAND / CODEX_CLI_COMMAND
# if your installed version uses different flags.
#
# Both are deliberately locked down to text-in / text-out:
#   * no tool use  - the CLI must not read the repository itself. All context comes
#     from the harness prompt, bounded by the retrieval budgets. If the CLI could
#     read files freely it would silently defeat the whole grounding layer.
#   * no writes    - patches are applied only by engineer-apply-patch behind the
#     human Apply gate. A CLI that can write would bypass that gate.
#   * temp cwd     - the subprocess runs outside the repository, so even a
#     misconfigured flag cannot reach project files.
# Defaults are argv LISTS, not shell strings. An empty argument like the one after
# --allowedTools cannot survive a round trip through shell-style quoting: PowerShell
# drops it, and shlex.split(..., posix=False) turns "" into a literal two-quote
# token. Building argv directly sidesteps both, and subprocess passes an empty
# string through correctly on Windows and POSIX alike.
AGENT_CLI_DEFAULT_ARGV: dict[str, list[str]] = {
    # NOTE: --permission-mode plan is deliberately NOT set. Plan mode steers the CLI
    # toward conversational planning prose, which breaks both callers: the harness
    # needs strict JSON and the baseline needs a unified diff. Writes are already
    # impossible here - tools are disabled and the process runs in an empty temp cwd -
    # so plan mode bought no safety and cost correctness.
    "Claude Code CLI": [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--allowedTools",
        "",
    ],
    # Codex streams JSONL events whose schema varies by version, so parsing stdout is
    # fragile. --output-last-message writes just the final assistant message to a file,
    # which is unambiguous. {OUTPUT_FILE} is substituted with a temp path at call time.
    "Codex CLI": [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--output-last-message",
        "{OUTPUT_FILE}",
    ],
}

OUTPUT_FILE_TOKEN = "{OUTPUT_FILE}"

AGENT_CLI_ENV_OVERRIDE: dict[str, str] = {
    "Claude Code CLI": "CLAUDE_CLI_COMMAND",
    "Codex CLI": "CODEX_CLI_COMMAND",
    "Cursor Agent CLI": "CURSOR_CLI_COMMAND",
}


def resolve_cursor_agent_bin() -> str | None:
    """Locate cursor-agent / agent headless binary (Windows .CMD included)."""
    candidates: list[str] = []
    for raw in (
        os.environ.get("CURSOR_EXEC_PATH"),
        os.environ.get("CURSOR_AGENT_BIN"),
    ):
        if raw and raw.strip():
            candidates.append(raw.strip())
    candidates.extend(
        [
            "cursor-agent",
            "agent",
            str(Path.home() / ".local" / "bin" / "cursor-agent"),
            str(Path.home() / ".local" / "bin" / "agent"),
            str(Path.home() / "AppData" / "Local" / "cursor-agent" / "cursor-agent.exe"),
            str(Path.home() / "AppData" / "Local" / "cursor-agent" / "cursor-agent.CMD"),
            str(Path.home() / "AppData" / "Local" / "cursor-agent" / "agent.exe"),
        ]
    )
    for raw in candidates:
        path = Path(raw)
        if path.is_file():
            return str(path)
        found = shutil.which(raw)
        if found:
            return found
    return None


def cursor_agent_cli_argv(model: str | None = None) -> list[str]:
    bin_path = resolve_cursor_agent_bin()
    if not bin_path:
        return []
    sandbox = (os.environ.get("CURSOR_EXEC_SANDBOX") or "").strip().lower()
    if sandbox not in {"enabled", "disabled"}:
        sandbox = "disabled" if os.name == "nt" else "enabled"
    model_name = (model or os.environ.get("CURSOR_CLI_MODEL") or "auto").strip() or "auto"
    return [
        bin_path,
        "-p",
        "--output-format",
        "text",
        "--mode",
        "ask",
        "--sandbox",
        sandbox,
        "--model",
        model_name,
        "--trust",
    ]


def agent_cli_argv(provider_name: str) -> list[str]:
    """Resolve the argv for an agent CLI provider.

    An env override is parsed with POSIX quoting rules on every platform, because
    POSIX rules are the only ones that turn '""' into a genuine empty string. Native
    Windows quoting is applied later by subprocess when it rebuilds the command line.
    """
    if provider_name == "Cursor Agent CLI":
        return cursor_agent_cli_argv()
    override = os.environ.get(AGENT_CLI_ENV_OVERRIDE.get(provider_name, ""), "").strip()
    if override:
        try:
            return shlex.split(override, posix=True)
        except ValueError:
            return []
    return list(AGENT_CLI_DEFAULT_ARGV.get(provider_name, []))


def resolve_agent_cli_model_label(provider_name: str) -> str:
    """Best-effort label for the model an agent-CLI route will actually invoke.

    Scans the same argv agent_cli_argv() would actually spawn, so this stays
    correct automatically if that resolution logic changes. Cursor Agent CLI
    always threads an explicit --model flag (see cursor_agent_cli_argv). Claude
    Code CLI and Codex CLI only do when their *_CLI_COMMAND override sets
    -m/--model explicitly; otherwise the installed CLI's own default model runs,
    which CompanyBrain does not control and must never report as a confirmed
    value - the caller (resilient_calls) uses this instead of the registry's
    static model_id for its route/cost log.
    """
    argv = agent_cli_argv(provider_name)
    for flag in ("--model", "-m"):
        if flag in argv:
            idx = argv.index(flag)
            if idx + 1 < len(argv):
                return argv[idx + 1]
    return "unresolved (CLI default, not pinned by CompanyBrain)"



# 300s was the original floor. The bounded grounded-replan follow-up round (see
# app/engineer/replan.py) makes a second lead-model call with an evidence pack
# grown by the follow-up retrieval; live runs of that path timed out at 300s twice
# in a row and succeeded once 600s was given, so 300s is no longer a safe floor for
# every CLI-based call this harness makes. Raising the default (still env-overridable)
# only widens how long we wait before a genuine failure - it changes no fail-closed
# safety semantics.
AGENT_CLI_TIMEOUT_SECONDS = int(os.environ.get("AGENT_CLI_TIMEOUT_SECONDS", "600"))

# Some CLIs report exact token usage and real cost in their JSON envelope. When they
# do, that beats the character-based estimate the ledger falls back to. Populated by
# the most recent call; read it immediately after call_provider returns.
LAST_CALL_USAGE: dict = {}


def _reset_last_call_usage() -> None:
    LAST_CALL_USAGE.clear()


def _capture_cli_usage(envelope: dict) -> None:
    """Record exact usage from a CLI JSON envelope when the shape is recognised."""
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return
    prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("output_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return
    cached = usage.get("cache_read_input_tokens")
    LAST_CALL_USAGE.update(
        {
            "exact": True,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached if isinstance(cached, int) else 0,
            "reported_cost_usd": envelope.get("total_cost_usd"),
        }
    )


def _cli_error_detail(stdout: str, stderr: str) -> str:
    """Prefer the CLI's own human-readable error over a raw JSON dump."""
    for blob in (stdout or "", stderr or ""):
        blob = blob.strip()
        if not blob.startswith("{"):
            continue
        try:
            envelope = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(envelope, dict):
            for key in ("result", "error", "message"):
                value = envelope.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ((stderr or "") or (stdout or "")).strip()


def agent_cli_available(provider_name: str) -> bool:
    """True when the CLI binary for this provider is on PATH."""
    if provider_name == "Cursor Agent CLI":
        return resolve_cursor_agent_bin() is not None
    argv = agent_cli_argv(provider_name)
    if not argv:
        return False
    return shutil.which(argv[0]) is not None


def _extract_agent_cli_text(stdout: str) -> str:
    """Pull the final assistant message out of whatever shape the CLI emitted.

    Handles three shapes without needing per-CLI configuration:
      1. a single JSON object with a "result" field (claude -p --output-format json)
      2. newline-delimited JSON events (codex exec --json, claude stream-json)
      3. plain text
    """
    text = stdout.strip()
    if not text:
        return ""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("result", "text", "content", "message", "last_message"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    collected: list[str] = []
    saw_json_line = False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        saw_json_line = True
        if not isinstance(event, dict):
            continue
        for key in ("result", "last_message", "text", "delta"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                collected.append(value)
        message = event.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                collected.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        collected.append(block["text"])
    if collected:
        return collected[-1].strip() if len(collected) == 1 else "\n".join(collected).strip()
    if saw_json_line:
        return ""
    return text


def _call_agent_cli(
    provider: ProviderConfig,
    system: str,
    prompt: str,
    *,
    request_timeout: int | None = None,
) -> str:
    """Drive a locally installed agent CLI as a plain text-in/text-out model.

    The prompt is passed on stdin and the process is launched with shell=False, so
    repository content in the prompt can never be interpreted as shell syntax. The
    working directory is a throwaway temp dir so the CLI cannot reach the project
    even if its sandbox flags are misconfigured.
    """
    argv = agent_cli_argv(provider.name)
    if not argv:
        raise ProviderError(f"No CLI command configured for {provider.name}.", provider=provider.name)
    if shutil.which(argv[0]) is None:
        raise ProviderError(
            f"{argv[0]} is not on PATH. Install the CLI or set "
            f"{AGENT_CLI_ENV_OVERRIDE.get(provider.name, 'the CLI command env var')}.",
            provider=provider.name,
        )

    payload = f"{system}\n\n{prompt}" if system else prompt
    if provider.name == "Cursor Agent CLI":
        payload = (
            payload.replace("\u2192", "->")
            .replace("\u2190", "<-")
            .replace("\u2014", "-")
            .replace("\u2013", "-")
        )
    # Callers pass HTTP-calibrated timeouts (~45s). A CLI pays process startup plus
    # agent-loop overhead on top of generation, so treat the caller's value as a
    # floor, never a ceiling, or long patch generations get killed mid-flight.
    timeout = max(int(request_timeout or 0), AGENT_CLI_TIMEOUT_SECONDS)
    started = time.monotonic()
    _reset_last_call_usage()

    with tempfile.TemporaryDirectory(prefix="companybrain_cli_") as workdir:
        output_file = os.path.join(workdir, "last_message.txt")
        argv = [output_file if part == OUTPUT_FILE_TOKEN else part for part in argv]
        try:
            completed = subprocess.run(  # noqa: S603 - shell=False, argv is operator-configured
                argv,
                input=payload,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=workdir,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(
                f"{provider.name} timed out after {timeout}s.",
                provider=provider.name,
                retryable=True,
            ) from exc
        except OSError as exc:
            raise ProviderError(f"{provider.name} failed to start: {exc}", provider=provider.name) from exc

        # Read the last-message file before the temp dir is torn down.
        last_message = ""
        try:
            if os.path.isfile(output_file):
                with open(output_file, encoding="utf-8", errors="replace") as handle:
                    last_message = handle.read().strip()
        except OSError:
            last_message = ""

    if completed.returncode != 0:
        detail = _cli_error_detail(completed.stdout, completed.stderr)[:600]
        raise ProviderError(
            f"{provider.name} exited {completed.returncode}: {detail}",
            provider=provider.name,
            retryable=completed.returncode not in {1, 2},
        )

    stdout = completed.stdout or ""
    try:
        envelope = json.loads(stdout.strip())
    except json.JSONDecodeError:
        envelope = None
    if isinstance(envelope, dict):
        _capture_cli_usage(envelope)
        if envelope.get("is_error"):
            detail = _cli_error_detail(stdout, completed.stderr)[:600]
            raise ProviderError(
                f"{provider.name} reported an error: {detail}",
                provider=provider.name,
                retryable=False,
            )

    content = last_message or _extract_agent_cli_text(stdout)
    if not content:
        # Include bounded raw output. "No usable text" without showing what WAS
        # returned is undiagnosable and costs another round trip to investigate.
        stdout_excerpt = stdout.strip()[:500].replace("\n", " | ")
        stderr_excerpt = (completed.stderr or "").strip()[:300].replace("\n", " | ")
        raise ProviderError(
            f"{provider.name} returned no usable text after {round(time.monotonic() - started, 1)}s. "
            f"stdout[{len(stdout)} chars]: {stdout_excerpt or '(empty)'} :: "
            f"stderr: {stderr_excerpt or '(empty)'}",
            provider=provider.name,
            retryable=True,
        )
    return content


def disabled_providers() -> set[str]:
    raw = os.environ.get("DISABLED_PROVIDERS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def provider_status() -> list[tuple[str, str, str]]:
    rows = []
    disabled = disabled_providers()
    for provider in PROVIDERS.values():
        if provider.name in disabled:
            status = "disabled"
        elif provider.kind == "agent_cli":
            status = "cli ready" if agent_cli_available(provider.name) else "cli not on PATH"
        elif provider.env_key is None:
            status = "local"
        elif os.environ.get(provider.env_key):
            status = "configured"
        else:
            status = f"missing {provider.env_key}"
        rows.append((provider.name, provider.model, status))
    return rows


def call_provider(
    provider_name: str,
    role: str,
    prompt: str,
    model_override: str | None = None,
    *,
    request_timeout: int | None = None,
    max_retries_override: int | None = None,
    max_tokens: int | None = None,
) -> str:
    provider = PROVIDERS[provider_name]
    if provider.name in disabled_providers():
        raise ProviderError(f"{provider.name} is disabled by DISABLED_PROVIDERS.")
    system = role_system_prompt(role)

    if provider.kind == "openai_compatible":
        return _call_openai_compatible(
            provider,
            system,
            prompt,
            model_override=model_override,
            request_timeout=request_timeout,
            max_retries_override=max_retries_override,
            max_tokens=max_tokens,
        )
    if provider.kind == "agent_cli":
        return _call_agent_cli(provider, system, prompt, request_timeout=request_timeout)
    if provider.kind == "gemini_interactions":
        return _call_gemini(provider, system, prompt)
    if provider.kind == "anthropic":
        return _call_anthropic(
            provider,
            system,
            prompt,
            model_override=model_override,
            request_timeout=request_timeout,
            max_tokens=max_tokens,
        )
    raise ProviderError(f"Unsupported provider kind: {provider.kind}")


def role_system_prompt(role: str) -> str:
    return (
        f"You are the {role} lens inside a one-person Company Brain. "
        "Be concrete. Separate facts from assumptions. Identify risks and the smallest next action. "
        "Do not pretend to have external evidence unless it is provided."
    )


def _message_headers(headers: Message) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def _json_request(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: int = 90,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> JsonResponse:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return JsonResponse(
                body=json.loads(response.read().decode("utf-8", errors="replace")),
                headers=_message_headers(response.headers),
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        header_map = _message_headers(exc.headers)
        body_lower = body.lower()
        quota_exhausted = "insufficient balance" in body_lower or "exceeded_current_quota" in body_lower
        raise ProviderError(
            f"HTTP {exc.code}: {body[:1200]}",
            status_code=exc.code,
            retry_after=retry_after_seconds(header_map),
            headers=header_map,
            retryable=(exc.code in {408, 409, 429, 500, 502, 503, 504}) and not quota_exhausted,
            model=model,
            provider=provider,
            quota_exhausted=quota_exhausted,
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(
            f"Network error: {exc}",
            retryable=True,
            model=model,
            provider=provider,
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ProviderError(
            f"Timeout: {exc}",
            retryable=True,
            model=model,
            provider=provider,
        ) from exc


def _call_openai_compatible(
    provider: ProviderConfig,
    system: str,
    prompt: str,
    model_override: str | None = None,
    *,
    request_timeout: int | None = None,
    max_retries_override: int | None = None,
    max_tokens: int | None = None,
) -> str:
    if not provider.env_key:
        raise ProviderError(f"{provider.name} needs an API key env var.")
    api_key = os.environ.get(provider.env_key)
    if not api_key:
        raise ProviderError(f"Set {provider.env_key} before calling {provider.name}.")

    base_url = os.environ.get(provider.env_base_url or "", provider.base_url or "").rstrip("/")
    model = model_override or provider.model
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    use_limiter = provider.name in {"NVIDIA NIM", "Kimi / Moonshot"}
    max_retries = max_retries_override if max_retries_override is not None else (1 if provider.name == "NVIDIA NIM" else 0)
    for attempt in range(max_retries + 1):
        try:
            if use_limiter:
                acquire_slot(provider.name, model)
            response = _json_request(
                f"{base_url}/chat/completions",
                payload,
                headers,
                timeout=request_timeout or 90,
                provider=provider.name,
                model=model,
            )
            result = response.body
            content = _extract_chat_content(result)
            if use_limiter:
                record_success(provider.name, model, response.headers)
            return content
            break
        except RateLimitError as exc:
            raise ProviderError(
                str(exc),
                retry_after=exc.retry_after,
                retryable=True,
                model=model,
                provider=provider.name,
            ) from exc
        except ProviderError as exc:
            if use_limiter:
                record_failure(provider.name, model, exc)
            if not exc.retryable or attempt >= max_retries or exc.is_rate_limit:
                raise
            time.sleep(min(exc.retry_after or 3, 15))
    raise ProviderError(f"{provider.name} call did not return a response.", retryable=True, model=model, provider=provider.name)


def _extract_chat_content(result: dict) -> str:
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            f"Unexpected response: {json.dumps(result)[:1200]}",
            retryable=True,
        ) from exc
    if content is None or content == "":
        raise ProviderError(
            f"Empty response content: {json.dumps(result)[:1200]}",
            retryable=True,
        )
    return content


def list_openai_compatible_models(provider_name: str) -> list[str]:
    provider = PROVIDERS[provider_name]
    if provider.kind != "openai_compatible":
        raise ProviderError(f"{provider_name} is not OpenAI-compatible.")
    if not provider.env_key:
        raise ProviderError(f"{provider.name} needs an API key env var.")
    api_key = os.environ.get(provider.env_key)
    if not api_key:
        raise ProviderError(f"Set {provider.env_key} before listing {provider.name} models.")
    base_url = os.environ.get(provider.env_base_url or "", provider.base_url or "").rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"HTTP {exc.code}: {body[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Network error: {exc}") from exc
    return sorted(item.get("id", "") for item in result.get("data", []) if item.get("id"))


def _call_gemini(provider: ProviderConfig, system: str, prompt: str) -> str:
    api_key = os.environ.get(provider.env_key or "")
    if not api_key:
        raise ProviderError("Set GEMINI_API_KEY before calling Gemini.")

    payload = {
        "model": provider.model,
        "system_instruction": system,
        "input": prompt,
    }
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
        "Api-Revision": "2026-05-20",
    }
    try:
        acquire_slot(provider.name, provider.model)
        response = _json_request(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            payload,
            headers,
            provider=provider.name,
            model=provider.model,
        )
        result = response.body
    except RateLimitError as exc:
        raise ProviderError(
            str(exc),
            retry_after=exc.retry_after,
            retryable=True,
            model=provider.model,
            provider=provider.name,
        ) from exc
    except ProviderError as exc:
        record_failure(provider.name, provider.model, exc)
        raise
    if "output_text" in result:
        record_success(provider.name, provider.model, response.headers)
        return result["output_text"]
    try:
        parts = []
        for step in result.get("steps", []):
            for item in step.get("content", []):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
        if parts:
            record_success(provider.name, provider.model, response.headers)
            return "\n".join(parts)
    except AttributeError:
        pass
    exc = ProviderError(f"Unexpected Gemini response: {json.dumps(result)[:1200]}", retryable=True, model=provider.model, provider=provider.name)
    record_failure(provider.name, provider.model, exc)
    raise exc


def _call_anthropic(
    provider: ProviderConfig,
    system: str,
    prompt: str,
    model_override: str | None = None,
    *,
    request_timeout: int | None = None,
    max_tokens: int | None = None,
) -> str:
    if not provider.env_key:
        raise ProviderError(f"{provider.name} needs an API key env var.")
    api_key = os.environ.get(provider.env_key)
    if not api_key:
        raise ProviderError(f"Set {provider.env_key} before calling {provider.name}.")

    base_url = os.environ.get(provider.env_base_url or "", provider.base_url or "").rstrip("/")
    model = model_override or provider.model
    payload = {
        "model": model,
        "max_tokens": max_tokens if max_tokens is not None else 4096,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    response = _json_request(
        f"{base_url}/messages",
        payload,
        headers,
        timeout=request_timeout or 90,
        provider=provider.name,
        model=model,
    )
    return _extract_anthropic_content(response.body, model=model, provider=provider.name)


def _extract_anthropic_content(result: dict, *, model: str, provider: str) -> str:
    try:
        blocks = result["content"]
    except (KeyError, TypeError) as exc:
        raise ProviderError(
            f"Unexpected Anthropic response: {json.dumps(result)[:1200]}",
            retryable=True,
            model=model,
            provider=provider,
        ) from exc
    parts = []
    for block in blocks or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    text = "".join(parts)
    if not text:
        raise ProviderError(
            f"Empty Anthropic response content: {json.dumps(result)[:1200]}",
            retryable=True,
            model=model,
            provider=provider,
        )
    return text
