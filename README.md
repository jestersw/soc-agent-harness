# SOC Agent Harness

An LLM agent that triages security alerts: it explores an evidence graph through
runtime tools and submits a verdict (`malicious` or `benign`) with incident-response
artifacts. Built for the BlueSec 2026 agent tournament.

The model is a cheap, fast one (DeepSeek V4 Flash). Most of the score came from
the harness around it, not from the model.

## Results

| Version | Arena | Score | Notes |
|---|---|---|---|
| v7 | practice | 10.81 | baseline harness |
| v8 | practice | 13.69 | verdict rules against living-off-the-land attacks |
| v8 | tournament | **19.52** | 40/40 tasks, 4 parallel workers |
| v9.1 | tournament | 17.02 | v8 + admin-activity and brute-force rules |

Cost: about $0,01-0.05 per task, roughly $2 for the whole tournament day.

## How it works
alert -> auto search on the alert process
-> loop: LLM picks the next tool call (structured JSON)
-> harness validates the call (ids, duplicates, budget)
-> runtime tool -> compact feedback back to the LLM
-> finish_investigation (verdict + artifacts or legitimacy evidence)

## Key engineering decisions

- **ID grounding.** `IdIndex` tracks every entity, relation and AD object id seen in
  the alert and tool results. Calls with invented ids are blocked locally or
  rerouted to `search`, so the agent does not pay runtime penalties for
  hallucinated ids. The final submission is sanitized against the same index.
- **Robust structured output.** The client rotates `json_schema`, `json_object` and
  plain modes between retries, extracts JSON from noisy replies, and parses
  DeepSeek's native DSML tool-call format when the model ignores the schema.
- **Lean history.** Tool feedback contains only the keys that changed in the
  observation, which cut prompt growth from ~7k tokens per step to a few hundred.
- **Loop control.** Duplicate calls are blocked, repeated blocked steps force a new
  search, and the agent is warned when its step budget is running out.
- **Parallel workers.** `BLUESEC_AGENT_WORKERS` runs several investigations inside
  one run. Leasing is serialized with a lock and retries when the server's task
  capacity is reached. Four workers gave about a 3x speedup.
- **Fault tolerance.** Transport errors and 5xx/429 from the LLM provider are
  retried with backoff; a failed task abort no longer kills the run. This kept
  the run alive through a runtime outage during the tournament.
- **Prompt addenda.** `BLUESEC_PROMPT_EXTRA` appends rules to the system prompt
  without code changes, which made A/B runs cheap.

## What did not work

- **Admin-activity rule (v9.1).** Treating privileged WinRM/PsExec/GPO activity
  without concrete attack signals as benign scored well on an 18-task practice
  sample, but the tournament run with it scored lower (17.02 vs 19.52). Each
  tournament run gets its own task set, so rule quality and task mix cannot be
  fully separated. Correct benign verdicts earn few points, while missed attacks
  cost a lot, so "malicious when in doubt" held up better.
- **Shorter model outputs** to speed up steps gave only ~6% and were reverted.
- **A newer model version** (v4.1 flash) cost twice as much with no score gain.

## Running

```bash
uv sync
cp .env.example .env
uv run --env-file .env bluesec1-agent
```

Fill in the runtime endpoint and token from the competition site and an
OpenAI-compatible LLM provider. Optional settings:

| Variable | Purpose |
|---|---|
| `LLM_RESPONSE_FORMAT` | `json_object` or `json_schema` as the primary mode |
| `BLUESEC_AGENT_WORKERS` | parallel investigations per run |
| `BLUESEC_AGENT_MAX_STEPS` | step budget per task |
| `BLUESEC_AGENT_MAX_TASKS` | stop after N tasks, useful for quick tests |
| `BLUESEC_PROMPT_EXTRA` | extra rules appended to the system prompt |

Per-task traces are written to `traces/<task_id>.jsonl`.

## Attribution

Based on the starter template provided by the BlueSec 2026 organizers
(False-Positive-Community/bluesec1-agent). The runtime client in
`src/bluesec1_client` and the proto definitions are their code; the agent
harness changes in `src/bluesec1_agent` are mine.
