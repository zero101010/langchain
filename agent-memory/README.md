# agent-memory — how an agent's "memory" actually works

A tiny LangChain agent that explores a repository with three tools and prints
its own **message list** so you can *see* what "memory" means for an agent.

The point: an agent has no hidden state. Its only memory is the growing list of
messages, which is re-sent **in full** to the model on every step.

## The agent

`repo-agent.py` gives the model three stdlib-backed tools and nothing else — it
knows nothing about the target repo up front and must discover everything:

| Tool | What it does |
|------|--------------|
| `list_files(subdir)` | List files/folders (its way to look around) |
| `search_files(query)` | grep a literal string across the repo |
| `read_file(path)` | Read one file (capped at 8k chars) |

The model is **Claude Haiku 4.5** served through the [TrueFoundry](https://www.truefoundry.com)
gateway (OpenAI-compatible), so the code uses `ChatOpenAI` pointed at the gateway.

## Setup

Uses the venv from the sibling `learning/` project (same deps).

Copy the env template and fill in your keys:

```bash
cp .env.example .env
```

Required in `.env`:

```
TRUEFOUNDRY_API_KEY=<your TrueFoundry PAT>
TRUEFOUNDRY_BASE_URL=https://gateway.truefoundry.ai/api/llm
TFY_METADATA={"team":"ops-ai"}   # sent as the X-TFY-Metadata header; the gateway requires a "team" tag
```

## Run

Point `REPO_ROOT` at any repo you want the agent to explore:

```bash
REPO_ROOT=../fix-pipeline-issue LANGSMITH_TRACING=false ../learning/.venv/bin/python repo-agent.py
```

- `REPO_ROOT` — the folder the agent is allowed to explore (defaults to `.`).
- `LANGSMITH_TRACING=false` — silences LangSmith logging (optional).

Edit the `question` at the bottom of `repo-agent.py` to ask something else.

## What you'll see (the memory)

The agent runs a loop — model picks a tool, your Python runs it, the result is
appended to the list — until the model answers with plain text. At the end it
prints the whole accumulated `messages` list:

```
[0] HumanMessage:  "What is the goal of this repo?"      ← your question
[1] AIMessage:     [wants to call] list_files({...})     ← model decides step 1
[2] ToolMessage:   "agent-suggestion-pipeline.py\n..."   ← tool result appended
[3] AIMessage:     [wants to call] read_file({...})
[4] ToolMessage:   "<file contents>"
...
[N] AIMessage:     "<final answer>"                       ← no tool call → done
```

**That list is the memory.** The entire thing is re-sent to the model on every
step — the model is stateless between calls, so the only reason it "remembers"
what it already saw is that those messages are physically back in the prompt.
Each tool result therefore also gets re-sent (and re-billed) on every later
call, which is why `read_file` is capped and why long chats need summarization.
