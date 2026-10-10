import asyncio
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from bluesec1_agent.llm import StructuredStepClient
from bluesec1_client import (
    RemoteBenchmarkClient,
    RemoteBenchmarkSession,
    TaskResult,
    ToolCallResult,
)

SYSTEM_PROMPT = """
You are a senior SOC analyst investigating one security alert in an evidence graph.
Goal: reconstruct the incident chain with as few tool calls as possible, then submit
a justified verdict. Every extra call lowers the score; a typical case needs 6-12 calls.

Return exactly one JSON object matching the NextStep schema on every turn.

How to investigate:
1. Read the alert carefully. Note ids, hashes, paths, users, logon_id, session_id,
   parent process, timestamps. A missing or null field means "unknown", not "absent".
2. `search` with an exact entity id returns the entity AND all its relations in one
   call. Prefer it over get_entity followed by several get_relation calls.
3. `search` with a hash, IP or unique string finds matching objects across the graph.
   The query is ONE exact token copied from the data: an id, hash, IP, file name or
   username. Never add words like "pid" or combine several terms. Leave scope and
   time_window empty unless a previous search returned too many results. If a search
   returns nothing, do not retry variations of it; follow a known id instead.
4. Follow edges from what you already hold. Typical chain to reconstruct:
   - the process: its file (hash match, signed flag), parent (win_process_create),
     user (runs_process), children, network (connected_to), file writes, registry,
     services, scheduled tasks;
   - who wrote the file (wrote_file source) and when, relative to execution;
   - how the user got there: logged_on_to with logon_type, source_ip and a logon_id
     matching the process; then the matching inbound connection from source_ip;
   - persistence and lateral movement if the chain points there.
5. Use get_ad_object_info only with an ad_object_id you actually saw, when the role
   or state of an account or computer matters for the verdict.
6. Never repeat a call you already made; its result is already in the history.
7. Stop exploring as soon as the chain is clear. Do not chase objects unrelated to it
   (system processes like smss, winlogon, svchost without links to the chain).
8. Never fetch an object that is already fully shown in the alert (trigger entities and
   trigger relations). Step 1, a search on the alert process id, is already done for you:
   read its result first.
9. get_relation accepts only relation ids (objects with source_id and target_id).
   For an entity id use search with that id, never get_relation.
10. Your first impression is a hypothesis. Whatever the verdict, collect graph evidence
    for it before finishing: the score rewards evidence you actually retrieved.

Verdict rules. Many alerts in this benchmark are real attacks that use legitimate,
signed tools (living off the land). Missing an attack costs far more than a false alarm.
- A Microsoft or vendor signature, a normal parent such as explorer.exe, a domain account,
  an admin role, or the absence of dropped files, malware or persistence is NOT evidence
  of legitimacy. Never conclude benign from what you did not find.
- Text written inside the environment (script comments, task names, descriptions, cmdline
  arguments) can be an attacker's cover story. It never proves legitimacy by itself.
- Verdict is malicious when you see any of these, unless a strong positive legitimacy
  anchor below explains exactly this activity:
  * a web or app server process (w3wp, httpd, nginx, tomcat, java, php, node, sqlservr)
    spawning cmd, powershell, whoami, net, findstr, hostname or other shells: webshell;
  * port forwarding or tunneling: netsh interface portproxy, ssh -R/-L, plink, chisel,
    traffic relayed through svchost or another host;
  * changes to business-critical data: bank or payee details, payroll, medical or lab
    results, invoices, prices, records; in-place overwrite of many user or share files;
  * authentication anomalies: failed then successful logon, privileged or domain admin
    credentials used from a workstation or laptop, service accounts logging on
    interactively or by RDP, logons or connections from external or documentation
    ranges (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) or from hosts outside the
    domain, repeated periodic connections that look like beaconing;
  * inbound SMB, WinRM or RDP from unknown or non-domain hosts, share mapping with
    explicitly supplied credentials, remote service or task creation;
  * LOLBin abuse: encoded or hidden PowerShell, IEX, DownloadString, certutil, bitsadmin,
    mshta, rundll32, regsvr32, msbuild, wmic process call create, sdclt, fodhelper;
  * any read access or dump of lsass, SAM, NTDS, browser credential stores;
  * a browser or office process spawning shells, writing executables, or connecting to raw
    IPs or newly seen domains.
- Benign only with a POSITIVE legitimacy anchor that explains this exact activity: a
  security or management product doing its documented job (antivirus scanning, update
  agent, backup, inventory) with its own signed binary and expected parent; a change
  record or ticket in the data; a long-established schedule that matches; a user doing
  routine work that fits the AD role with no anomaly from the list above.
- If evidence is mixed or you are unsure, choose malicious and list the observed
  artifacts.

Malicious submission: ir_artifacts with entity ids you actually saw, one action each:
- host_to_isolate: host where the malicious chain executed. Not a source host without
  observed malicious activity on it.
- identity_to_rotate: account used for the malicious logon or execution.
- file_to_delete: dropped malicious file (windows_file or linux_file entity).
- network_block: the network_connection entity to the external destination.
- persistence_to_remove: service, scheduled task, run key, WMI binding, systemd unit.
- process_observed, file_observed, registry_observed, identity_observed: evidence
  objects worth recording that need no containment action.
Benign submission: legitimacy_evidence with 3-6 items, each citing an entity or relation
id you retrieved plus the exact property field names that prove legitimacy.
Order of evidence:
1. FIRST the relations that made the activity look suspicious (opened_handle or
   accessed_process to lsass, connected_to, wrote_file, created_service, set_registry_value,
   win_process_create of the alert process). Cite each such relation with the inner field
   names that show it is harmless, e.g. ["granted_access"], ["dst_ip","dst_port"].
   Field names come from properties, access_properties or logon_properties of the relation.
2. Then the alert process itself with fields such as ["image_path","digital_signature"]
   or ["signed","signer"], whichever exist on that entity.
3. Then the parent process or account that explains the context.
Cite only fields that actually exist on that object. Do not cite unrelated objects such as
a SYSTEM service logon or the target process when the relation already proves the point.

reasoning in finish_investigation: a short factual chain with times and ids, plus what
is NOT known. No speculation presented as fact.
""".strip()

BUDGET_WARNING_STEPS = 3
STUCK_LIMIT = 3
TRACE_DIR = Path("traces")


def system_prompt() -> str:
    extra = os.environ.get("BLUESEC_PROMPT_EXTRA", "").strip()
    return f"{SYSTEM_PROMPT}\n\n{extra}" if extra else SYSTEM_PROMPT


def render_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)


class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    successful_tasks: int
    failed_tasks: int
    task_results: list[TaskResult] = Field(default_factory=list)


def compact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            packed = compact(item)
            if packed is None or packed == [] or packed == {}:
                continue
            result[key] = packed
        return result
    if isinstance(value, list):
        return [compact(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("[", "{"):
            try:
                return compact(json.loads(stripped))
            except json.JSONDecodeError:
                return value
    return value


def call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    return render_json(
        {"tool": tool_name, "arguments": {k: v for k, v in arguments.items() if k != "reasoning"}}
    )


def first_search_target(alert: dict[str, Any]) -> str | None:
    alert_text = alert.get("alert_text")
    if not isinstance(alert_text, dict):
        return None
    entities = [e for e in alert_text.get("trigger_entities", []) if isinstance(e, dict)]
    for entity in entities:
        if "process" in str(entity.get("type", "")) and entity.get("id"):
            return str(entity["id"])
    for entity in entities:
        if entity.get("type") != "host" and entity.get("id"):
            return str(entity["id"])
    return None


class IdIndex:
    def __init__(self) -> None:
        self.entities: set[str] = set()
        self.relations: set[str] = set()
        self.ad_objects: set[str] = set()
        self.seen: set[str] = set()
        self.fields: dict[str, set[str]] = {}

    def add(self, value: Any) -> None:
        if isinstance(value, dict):
            object_id = value.get("id")
            if isinstance(object_id, str):
                names = self.fields.setdefault(object_id, set())
                groups = ("properties", "access_properties", "logon_properties")
                for group in groups:
                    if isinstance(value.get(group), dict):
                        names.update(value[group].keys())
                if "source_id" not in value and not any(g in value for g in groups):
                    names.update(key for key in value if key != "id")
                if "source_id" in value or value.get("kind") == "relation":
                    self.relations.add(object_id)
                elif value.get("kind") in {"account", "computer", "group", "ds_object"}:
                    self.ad_objects.add(object_id)
                else:
                    self.entities.add(object_id)
            for key, item in value.items():
                if key in {"source_id", "target_id", "host_id"} and isinstance(item, str):
                    self.entities.add(item)
                if "ad_object_id" in key or key.endswith("_ids") or key == "manager_id":
                    for ad_id in item if isinstance(item, list) else [item]:
                        if isinstance(ad_id, str):
                            self.ad_objects.add(ad_id)
                if isinstance(item, str):
                    self.seen.add(item)
                if key == "relation_ids" and isinstance(item, dict):
                    for group in item.values():
                        for element in group if isinstance(group, list) else []:
                            if isinstance(element, str):
                                self.relations.add(element)
                                self.seen.add(element)
                if isinstance(item, list):
                    for element in item:
                        if isinstance(element, str):
                            self.seen.add(element)
                            if "relation" in key:
                                self.relations.add(element)
                self.add(item)
        elif isinstance(value, list):
            for item in value:
                self.add(item)

    def check(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        if tool_name == "get_relation":
            relation_id = str(arguments.get("relation_id", "")).strip()
            if not relation_id:
                return "relation_id is empty. Use an id of a relation you have seen."
            if relation_id in self.entities and relation_id not in self.relations:
                return (
                    f"{relation_id} is an entity id, not a relation id. "
                    "Use search with this id to get the entity and all its relations."
                )
            if relation_id not in self.relations and relation_id not in self.seen:
                return f"{relation_id} never appeared in the alert or tool results."
        if tool_name == "get_entity":
            entity_id = str(arguments.get("entity_id", "")).strip()
            if not entity_id:
                return "entity_id is empty. Use an id of an entity you have seen."
            if entity_id in self.relations and entity_id not in self.entities:
                return f"{entity_id} is a relation id. Use get_relation with it."
            if entity_id not in self.entities and entity_id not in self.seen:
                return f"{entity_id} never appeared in the alert or tool results."
        if tool_name == "get_ad_object_info":
            ad_object_id = str(arguments.get("ad_object_id", "")).strip()
            if not ad_object_id:
                return "ad_object_id is empty. Use an ad_object_id you have seen."
            if ad_object_id not in self.ad_objects and ad_object_id not in self.seen:
                return f"{ad_object_id} never appeared as an AD object id."
        return None

    def sanitize_submission(self, arguments: dict[str, Any]) -> list[str]:
        submission = arguments.get("submission")
        if not isinstance(submission, dict):
            return []
        notes: list[str] = []
        artifacts = submission.get("ir_artifacts")
        if isinstance(artifacts, list):
            kept = [a for a in artifacts if a.get("entity_id") in self.seen | self.entities]
            if len(kept) != len(artifacts):
                notes.append(f"dropped {len(artifacts) - len(kept)} unknown artifact ids")
            submission["ir_artifacts"] = kept
        evidence = submission.get("legitimacy_evidence")
        if isinstance(evidence, list):
            cleaned = []
            for item in evidence:
                object_id = item.get("entity_id") or item.get("relation_id")
                if object_id not in self.seen | self.entities | self.relations:
                    notes.append(f"dropped unknown evidence {object_id}")
                    continue
                known = self.fields.get(object_id)
                fields = list(item.get("property_fields") or [])
                if known is not None:
                    valid = [name for name in fields if name in known]
                    if len(valid) != len(fields):
                        dropped = sorted(set(fields) - set(valid))
                        notes.append(f"{str(object_id)[:8]}: dropped fields {dropped}")
                    fields = valid
                if item.get("anchor") == "entity" and not fields:
                    notes.append(
                        f"dropped entity evidence {str(object_id)[:8]} without valid fields"
                    )
                    continue
                cleaned.append({**item, "property_fields": fields})
            submission["legitimacy_evidence"] = cleaned
        return notes

    def reroute(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        key = {"get_relation": "relation_id", "get_entity": "entity_id"}.get(tool_name)
        if key is None:
            return None
        value = str(arguments.get(key, "")).strip()
        if len(value) < 5 or value in self.relations:
            return None
        if tool_name == "get_entity" and value in self.entities:
            return None
        if value in self.entities or value in self.seen:
            return value
        return None


class SimpleSGRAgent:
    def __init__(self, llm: StructuredStepClient, *, max_steps: int = 100) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.llm = llm
        self.max_steps = max_steps

    async def run(self, client: RemoteBenchmarkClient) -> AgentRunResult:
        remote_run = client.run
        if remote_run is None:
            raise RuntimeError("RemoteBenchmarkClient must be open before running the agent.")

        task_results: list[TaskResult] = []
        max_tasks = int(os.environ.get("BLUESEC_AGENT_MAX_TASKS", "0"))
        workers = max(1, int(os.environ.get("BLUESEC_AGENT_WORKERS", "1")))
        leased = 0
        lease_lock = asyncio.Lock()

        async def lease() -> RemoteBenchmarkSession | None:
            nonlocal leased
            async with lease_lock:
                if max_tasks and leased >= max_tasks:
                    return None
                for attempt in range(90):
                    try:
                        session = await client.start_task()
                        break
                    except Exception as exc:
                        logger.warning("Lease limited, retrying: {}", exc)
                        await asyncio.sleep(min(15, 5 + attempt))
                else:
                    return None
                if session is not None:
                    leased += 1
                return session

        async def worker(number: int) -> None:
            while (session := await lease()) is not None:
                logger.info("Worker {} starts task {}.", number, session.task_id)
                try:
                    task_result = await self._investigate(session)
                except Exception as exc:
                    logger.error("Task {} failed in the agent: {}", session.task_id, exc)
                    error_message = f"{type(exc).__name__}: {exc}"[:2_000]
                    try:
                        aborted_result = await session.abort_task(error_message=error_message)
                        task_result = TaskResult.model_validate(aborted_result)
                    except Exception as abort_exc:
                        logger.error("Abort of task {} failed: {}", session.task_id, abort_exc)
                        await asyncio.sleep(10)
                        continue
                task_results.append(task_result)
                logger.info("Finished tasks: {}.", len(task_results))

        await asyncio.gather(*(worker(number) for number in range(workers)))

        successful_tasks = sum(
            result.completion_reason in {"terminated", "truncated"} for result in task_results
        )
        return AgentRunResult(
            run_id=remote_run.run_id,
            successful_tasks=successful_tasks,
            failed_tasks=len(task_results) - successful_tasks,
            task_results=task_results,
        )

    async def _investigate(self, session: RemoteBenchmarkSession) -> TaskResult:
        observation = session.task.observation
        tool_specs = self._tool_specs(observation)
        available_tools = {
            specification["name"]
            for specification in tool_specs
            if isinstance(specification.get("name"), str)
        }
        if "finish_investigation" not in available_tools:
            raise ValueError("The runtime did not advertise finish_investigation.")

        alert = compact(
            {key: value for key, value in observation.items() if key != "available_tools"}
        )
        index = IdIndex()
        index.add(alert)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": render_json(
                    {
                        "instruction": "Investigate this alert and submit the correct response.",
                        "step_budget": self.max_steps,
                        "alert": alert,
                        "available_tools": compact(tool_specs),
                    }
                ),
            },
        ]
        seen_calls: set[str] = set()
        TRACE_DIR.mkdir(exist_ok=True)
        trace_path = TRACE_DIR / f"{session.task_id}.jsonl"
        self._trace(trace_path, {"step": 0, "alert": alert})

        step_index = 0
        blocked_in_row = 0
        first_target = first_search_target(alert)
        if first_target and "search" in available_tools:
            step_index = 1
            arguments = {
                "query": first_target,
                "reasoning": "Expand the alert process: parent, user, children, files, network.",
            }
            seen_calls.add(call_signature("search", arguments))
            response = await session.call_tool("search", arguments)
            feedback = self._tool_feedback(response, observation)
            index.add(feedback)
            logger.info("Step 1/{}: search (auto) -> {}.", self.max_steps, feedback.get("status"))
            self._append_call(
                messages, 1, "Expand the alert process.", "search", arguments, feedback
            )
            self._trace(
                trace_path,
                {"step": 1, "tool": "search", "arguments": arguments, "feedback": feedback},
            )

        while step_index < self.max_steps:
            step_index += 1
            remaining = self.max_steps - step_index + 1
            if remaining <= BUDGET_WARNING_STEPS:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Only {remaining} tool calls left. Call finish_investigation "
                            "now with the evidence you already have."
                        ),
                    }
                )

            if blocked_in_row >= STUCK_LIMIT:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your last {blocked_in_row} calls were rejected. Stop repeating "
                            "them. Either call search with a NEW query taken from the results "
                            "(an id, hash, IP or path), or call finish_investigation now."
                        ),
                    }
                )
                blocked_in_row = 0

            step = await self.llm.next_step(messages)
            tool_name = step.function.tool
            arguments = step.function.model_dump(
                mode="json",
                by_alias=True,
                exclude={"tool"},
                exclude_none=True,
            )
            note: str | None = None
            if tool_name in available_tools and "search" in available_tools:
                target = index.reroute(tool_name, arguments)
                if target is not None:
                    note = f"{tool_name} was rerouted to search for {target!r}."
                    logger.info("Rerouted {} to search for {}.", tool_name, target)
                    tool_name = "search"
                    arguments = {
                        "query": target,
                        "reasoning": str(arguments.get("reasoning", "Expand this object.")),
                    }
            call_key = call_signature(tool_name, arguments)
            reward: float | None = None

            if tool_name not in available_tools:
                feedback: dict[str, Any] = {
                    "status": "error",
                    "message": (
                        f"Tool {tool_name!r} is not advertised for this task. "
                        f"Available tools: {sorted(available_tools)}"
                    ),
                }
                logger.warning("Rejected unavailable tool {} locally.", tool_name)
            elif tool_name != "finish_investigation" and call_key in seen_calls:
                feedback = {
                    "status": "duplicate",
                    "message": (
                        "You already made this exact call. Its result is in the history. "
                        "Do not repeat it. To see relations of an entity use search with "
                        "its id; otherwise call finish_investigation."
                    ),
                }
                logger.warning("Blocked duplicate call {} locally.", tool_name)
            elif (problem := index.check(tool_name, arguments)) is not None:
                feedback = {"status": "invalid", "message": problem}
                logger.warning("Blocked invalid call {} locally: {}", tool_name, problem)
            else:
                if tool_name == "finish_investigation":
                    for fix in index.sanitize_submission(arguments):
                        logger.info("Sanitized submission: {}", fix)
                seen_calls.add(call_key)
                response = await session.call_tool(tool_name, arguments)
                feedback = self._tool_feedback(response, observation)
                reward = response.reward
                index.add(feedback)
                logger.info(
                    "Step {}/{}: {} -> {} (reward {}).",
                    step_index,
                    self.max_steps,
                    tool_name,
                    feedback.get("status", "unknown"),
                    response.reward,
                )
                blocked_in_row = 0
                if response.task_result is not None:
                    self._trace(
                        trace_path,
                        {
                            "step": step_index,
                            "state": step.current_state,
                            "tool": tool_name,
                            "arguments": arguments,
                            "feedback": feedback,
                            "task_result": response.task_result,
                        },
                    )
                    return TaskResult.model_validate(response.task_result)

            if note is not None:
                feedback = {"note": note, **feedback}
            if feedback.get("status") in {"duplicate", "invalid", "error"} and reward is None:
                blocked_in_row += 1
            content = render_json(feedback)
            logger.info("Step {} tool result size: {} chars.", step_index, len(content))
            self._trace(
                trace_path,
                {
                    "step": step_index,
                    "state": step.current_state,
                    "tool": tool_name,
                    "arguments": arguments,
                    "feedback": feedback,
                    "reward": reward,
                    "chars": len(content),
                },
            )
            self._append_call(
                messages,
                step_index,
                step.plan_remaining_steps_brief[0],
                tool_name,
                arguments,
                feedback,
            )

        raise RuntimeError(
            f"Agent hit the {self.max_steps}-step limit without finish_investigation."
        )

    @staticmethod
    def _append_call(
        messages: list[dict[str, Any]],
        step_index: int,
        note: str,
        tool_name: str,
        arguments: dict[str, Any],
        feedback: dict[str, Any],
    ) -> None:
        tool_call_id = f"step_{step_index}"
        messages.append(
            {
                "role": "assistant",
                "content": note,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": tool_call_id,
                        "function": {"name": tool_name, "arguments": render_json(arguments)},
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "content": render_json(feedback), "tool_call_id": tool_call_id}
        )

    @staticmethod
    def _trace(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as trace:
            trace.write(render_json(record) + "\n")

    @staticmethod
    def _tool_specs(observation: dict[str, Any]) -> list[dict[str, Any]]:
        raw_tool_specs = observation.get("available_tools")
        if not isinstance(raw_tool_specs, str):
            raise ValueError("Observation has no JSON-encoded available_tools field.")
        try:
            tool_specs = json.loads(raw_tool_specs)
        except json.JSONDecodeError as exc:
            raise ValueError("available_tools is not valid JSON.") from exc
        if not isinstance(tool_specs, list) or not all(
            isinstance(specification, dict) for specification in tool_specs
        ):
            raise ValueError("available_tools must encode a list of tool specifications.")
        return [dict(specification) for specification in tool_specs]

    @staticmethod
    def _tool_feedback(response: ToolCallResult, baseline: dict[str, Any]) -> dict[str, Any]:
        if "tool_result" in response.info:
            raw_result = response.info["tool_result"]
        else:
            raw_result = {
                key: value
                for key, value in response.observation.items()
                if key != "available_tools" and baseline.get(key) != value
            }
        feedback = {
            "status": response.info.get("tool_status", "unknown"),
            "result": compact(raw_result),
            "error": response.info.get("error"),
            "reward": response.reward or None,
        }
        return {key: value for key, value in feedback.items() if value is not None}
