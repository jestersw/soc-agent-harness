import asyncio
import contextlib
import json
import re
import time
from types import TracebackType
from typing import Any, Protocol, Self, cast

from loguru import logger
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from bluesec1_agent.models import (
    NEXT_STEP_RESPONSE_FORMAT,
    FinishInvestigationRequest,
    NextStep,
)

STRUCTURED_OUTPUT_INSTRUCTION = """
Every assistant reply must be exactly one JSON object with these top-level fields:
{
  "current_state": "short evidence-focused string",
  "plan_remaining_steps_brief": ["one to five short step strings"],
  "task_completed": false,
  "function": {
    "tool": "one advertised tool name",
    "...": "that tool's arguments placed directly in this function object"
  }
}

Never use top-level `action`, `parameters`, `tool_name`, or `arguments` fields.
Never wrap tool arguments in a nested `parameters` or `arguments` object.
For example, a call with one advertised argument is:
{"tool":"advertised_tool","argument_name":"value"}
Set task_completed=true only for a function whose tool is finish_investigation.
Return no prose, markdown, XML, comments, or keys outside the required object.
""".strip()

REPAIR_INSTRUCTION = (
    "Your previous reply violated the required NextStep contract. " + STRUCTURED_OUTPUT_INSTRUCTION
)


class NextStepRequestError(RuntimeError):
    """Raised when an LLM cannot produce a valid structured step."""


class StructuredStepClient(Protocol):
    """Narrow async contract consumed by the investigation loop."""

    async def next_step(self, messages: list[dict[str, Any]]) -> NextStep:
        """Generate and validate one investigation step."""
        ...


class OpenAIStructuredStepClient:
    """Portable structured-output client for OpenAI-compatible providers."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_completion_tokens: int,
        max_retries: int,
        client: AsyncOpenAI | None = None,
        response_format_mode: str = "json_schema",
    ) -> None:
        """Initialize an OpenAI-compatible structured step generator.

        Args:
            base_url: OpenAI-compatible API root.
            api_key: Provider credential.
            model: Provider model identifier.
            timeout_seconds: Timeout applied to each provider request.
            max_completion_tokens: Maximum generated tokens per attempt.
            max_retries: Maximum schema-generation attempts per step.
            client: Optional injected SDK client used by tests.
        """
        self.model = model
        self.response_format_mode = response_format_mode
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def __aenter__(self) -> Self:
        """Enter the async LLM client context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close an internally owned provider client."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the provider transport created by this wrapper."""
        if self._owns_client:
            await self._client.close()

    async def next_step(self, messages: list[dict[str, Any]]) -> NextStep:
        """Return one valid `NextStep`, repairing malformed replies when possible.

        Args:
            messages: Current investigation conversation.

        Returns:
            A locally validated structured investigation step.

        Raises:
            NextStepRequestError: If every provider reply violates the schema.
        """
        schema = json.dumps(
            NEXT_STEP_RESPONSE_FORMAT["json_schema"]["schema"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        instruction = (
            f"{STRUCTURED_OUTPUT_INSTRUCTION}\n\nThe JSON object must match this JSON Schema:\n"
            f"{schema}"
        )
        alternate = "json_schema" if self.response_format_mode == "json_object" else "json_object"
        modes = [self.response_format_mode, alternate, "none"]
        request_messages = list(messages)
        if request_messages and request_messages[0].get("role") == "system":
            first_message = dict(request_messages[0])
            first_message["content"] = f"{first_message.get('content', '')}\n\n{instruction}"
            request_messages[0] = first_message
        else:
            request_messages.insert(0, {"role": "system", "content": instruction})
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            mode = modes[(attempt - 1) % len(modes)]
            options: dict[str, Any] = {}
            if mode == "json_schema":
                options["response_format"] = NEXT_STEP_RESPONSE_FORMAT
            elif mode == "json_object":
                options["response_format"] = {"type": "json_object"}
            started_at = time.perf_counter()
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=cast(list[ChatCompletionMessageParam], request_messages),
                    max_completion_tokens=self.max_completion_tokens,
                    **options,
                )
            except (APIConnectionError, APIStatusError) as exc:
                if (
                    isinstance(exc, APIStatusError)
                    and exc.status_code < 500
                    and exc.status_code != 429
                ):
                    raise
                last_error = exc
                logger.warning(
                    "LLM request failed on attempt {}/{} (mode {}): {}",
                    attempt,
                    self.max_retries,
                    mode,
                    exc,
                )
                await asyncio.sleep(min(30, 5 * attempt))
                continue
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            raw_content = extract_raw_content(response)
            total_tokens = response_total_tokens(response)

            try:
                step = parse_next_step_payload(raw_content)
            except ValueError as exc:
                step = salvage_next_step(raw_content)
                if step is None:
                    last_error = exc
                    logger.warning(
                        "Invalid LLM response on attempt {}/{} (mode {}): {}",
                        attempt,
                        self.max_retries,
                        mode,
                        raw_content[:300] or str(exc),
                    )
                    if attempt == self.max_retries:
                        break
                    request_messages.append({"role": "user", "content": REPAIR_INSTRUCTION})
                    continue
                logger.info("Recovered a malformed LLM response on attempt {}.", attempt)

            logger.info(
                "LLM selected {} in {} ms using {} tokens (mode {}).",
                step.function.tool,
                latency_ms,
                total_tokens,
                mode,
            )
            return step

        message = str(last_error or ValueError("Unable to parse model response."))
        raise NextStepRequestError(message) from last_error


def extract_message_text(message: Any) -> str:
    """Normalize string and multipart assistant content into plain text.

    Args:
        message: Provider-specific chat completion message.

    Returns:
        Concatenated textual message content.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(str(part.get("text", "")))
        elif getattr(part, "type", None) == "text":
            text_parts.append(str(getattr(part, "text", "")))
    return "".join(text_parts)


def extract_raw_content(response: Any) -> str:
    """Extract the first assistant message from a completion response.

    Args:
        response: OpenAI-compatible chat completion response.

    Returns:
        Raw assistant text.

    Raises:
        ValueError: If the provider response is missing a usable message.
    """
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Model response is missing choices.")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("Model response is missing a message.")

    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError(f"Model refused structured output: {refusal}")
    return extract_message_text(message)


def extract_json_object(raw_content: str) -> str:
    """Remove common provider wrappers around a JSON object.

    Args:
        raw_content: Untrusted assistant response text.

    Returns:
        Candidate JSON object text for local validation.
    """
    stripped = raw_content.strip()
    if not stripped:
        return ""
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    return match.group(0) if match is not None else stripped


def parse_next_step_payload(raw_content: str) -> NextStep:
    """Validate one raw provider reply as a consistent `NextStep`.

    Args:
        raw_content: Assistant response containing a structured step.

    Returns:
        Parsed and validated step.

    Raises:
        ValueError: If JSON, schema, or completion semantics are invalid.
    """
    if not raw_content:
        raise ValueError("Model returned empty content for NextStep response.")

    normalized_content = extract_json_object(raw_content)
    try:
        step = NextStep.model_validate_json(normalized_content)
    except ValidationError as exc:
        raise ValueError("Model returned invalid NextStep JSON.") from exc

    is_finish = isinstance(step.function, FinishInvestigationRequest)
    if step.task_completed != is_finish:
        raise ValueError(
            "task_completed must be true exactly when finish_investigation is selected."
        )
    return step


def _candidate_objects(value: Any, depth: int = 0) -> list[dict[str, Any]]:
    if depth > 4:
        return []
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "function" in value or "tool" in value:
            found.append(value)
        for item in value.values():
            found.extend(_candidate_objects(item, depth + 1))
    elif isinstance(value, list):
        for item in value:
            found.extend(_candidate_objects(item, depth + 1))
    elif isinstance(value, str) and "{" in value:
        with contextlib.suppress(json.JSONDecodeError):
            found.extend(_candidate_objects(json.loads(extract_json_object(value)), depth + 1))
    return found


DSML_INVOKE = re.compile(r"invoke name=\"([a-z_]+)\"(.*?)(?:</[^>]*invoke>|$)", re.DOTALL)
DSML_PARAMETER = re.compile(r"parameter name=\"([a-z_]+)\"[^>]*>(.*?)</", re.DOTALL)


def _dsml_function(raw_content: str) -> dict[str, Any] | None:
    match = DSML_INVOKE.search(raw_content)
    if match is None:
        return None
    function: dict[str, Any] = {"tool": match.group(1)}
    for name, value in DSML_PARAMETER.findall(match.group(2)):
        text = value.strip()
        try:
            function[name] = json.loads(text) if text[:1] in ("{", "[") else text
        except json.JSONDecodeError:
            function[name] = text
    return function


def salvage_next_step(raw_content: str) -> NextStep | None:
    if not raw_content:
        return None
    if "invoke name=" in raw_content:
        function = _dsml_function(raw_content)
        if function is not None:
            function.setdefault("reasoning", "Recovered tool call.")
            if function["tool"] == "search":
                function.setdefault("page_id", 1)
                function.setdefault("scope", None)
                function.setdefault("time_window", None)
                function.setdefault("sort_order", "ascending")
            note = raw_content.split("<", 1)[0].strip() or "Recovered tool call."
            with contextlib.suppress(ValidationError):
                return NextStep.model_validate(
                    {
                        "current_state": note[:500],
                        "plan_remaining_steps_brief": ["Continue the investigation."],
                        "task_completed": function["tool"] == "finish_investigation",
                        "function": function,
                    }
                )
    try:
        root = json.loads(extract_json_object(raw_content))
    except json.JSONDecodeError:
        root = raw_content
    for candidate in _candidate_objects(root):
        payload = dict(candidate)
        if "function" not in payload:
            payload = {"function": payload}
        function = payload.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("tool"), str):
            continue
        plan = payload.get("plan_remaining_steps_brief")
        if not isinstance(plan, list) or not plan:
            plan = ["Continue the investigation."]
        payload["plan_remaining_steps_brief"] = [str(item) for item in plan[:5]]
        payload["current_state"] = str(payload.get("current_state") or "Recovered step.")
        payload["task_completed"] = function["tool"] == "finish_investigation"
        try:
            return NextStep.model_validate(
                {
                    key: payload[key]
                    for key in (
                        "current_state",
                        "plan_remaining_steps_brief",
                        "task_completed",
                        "function",
                    )
                }
            )
        except ValidationError:
            continue
    return None


def response_total_tokens(response: Any) -> int:
    """Read total token usage when the provider supplies it.

    Args:
        response: OpenAI-compatible chat completion response.

    Returns:
        Non-negative total token count, or zero when unavailable.
    """
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 0)
    return total_tokens if isinstance(total_tokens, int) and total_tokens >= 0 else 0


def render_json(value: Any) -> str:
    """Serialize participant-visible context for an LLM message.

    Args:
        value: Public JSON-compatible value.

    Returns:
        Stable readable JSON text.
    """
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
