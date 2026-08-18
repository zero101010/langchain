import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# The agent can only see files under this root. Anything it "knows" about the
# repo, it has to discover by CALLING a tool — nothing is pre-loaded.
REPO_ROOT = Path(os.getenv("REPO_ROOT", ".")).resolve()

# ponytail: no fancy ignore rules; skip the usual noise dirs and move on.
IGNORE = {".git", ".venv", "__pycache__", "node_modules"}


def _safe(rel: str) -> Path:
    """Resolve a repo-relative path and refuse anything escaping REPO_ROOT."""
    p = (REPO_ROOT / rel).resolve()
    if REPO_ROOT not in p.parents and p != REPO_ROOT:
        raise ValueError(f"Path {rel!r} is outside the repo root")
    return p


@tool
def list_files(subdir: str = ".") -> str:
    """
    Lists the files and folders of a repository directory.
    Use it to discover what exists before reading or searching.
    args:
        subdir: path relative to the repo root (default: root)
    """
    base = _safe(subdir)
    if not base.is_dir():
        return f"{subdir} is not a directory."
    out = []
    for entry in sorted(base.iterdir()):
        if entry.name in IGNORE:
            continue
        rel = entry.relative_to(REPO_ROOT)
        out.append(f"{rel}/" if entry.is_dir() else str(rel))
    return "\n".join(out) or "(empty)"


@tool
def read_file(path: str) -> str:
    """
    Reads the content of a repository file.
    Use it when you already know which file you want to inspect.
    args:
        path: path relative to the repo root
    """
    p = _safe(path)
    if not p.is_file():
        return f"{path} does not exist or is not a file."
    text = p.read_text(errors="replace")
    # ponytail: hard cap so one huge file can't blow the context window.
    return text[:8000] + ("\n...(truncated)" if len(text) > 8000 else "")


@tool
def search_files(query: str) -> str:
    """
    Searches for a string across all text files in the repository (grep-like).
    Use it to find WHERE something is defined/mentioned.
    args:
        query: literal text to search for
    """
    hits = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORE]
        for name in files:
            fp = Path(root) / name
            try:
                for i, line in enumerate(fp.read_text(errors="replace").splitlines(), 1):
                    if query in line:
                        rel = fp.relative_to(REPO_ROOT)
                        hits.append(f"{rel}:{i}: {line.strip()[:120]}")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable
    if not hits:
        return f"No results for {query!r}."
    return "\n".join(hits[:50])  # ponytail: cap 50, enough for learning


# Claude Haiku 4.5 via the TrueFoundry gateway (OpenAI-compatible endpoint).
# base_url defaults to TrueFoundry's OpenAI-compat path; override if yours differs.
model = ChatOpenAI(
    model="google-vertex/anthropic-claude-haiku-4-5",  # TrueFoundry provider/model name
    api_key=os.environ["TRUEFOUNDRY_API_KEY"],         # your TrueFoundry PAT
    base_url=os.getenv("TRUEFOUNDRY_BASE_URL", "https://gateway.truefoundry.ai/api/llm"),
    temperature=0,
    default_headers={"X-TFY-Metadata": os.getenv("TFY_METADATA", '{"team":"ops-ai"}')},  # tags requests in the gateway
)

agent = create_agent(
    model=model,
    system_prompt=(
        "You are an agent that explores a code repository. "
        "You do NOT know the repo beforehand: discover everything by calling the "
        "list_files, search_files and read_file tools. Explore step by step until you can answer."
    ),
    tools=[list_files, search_files, read_file],
)


if __name__ == "__main__":
    question = "Which agents exist in this repo and which LLM do they use?"

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # >>> HERE is the "memory": the message list that only grows. <<<
    # Every tool call and every result becomes a message in this list,
    # and the ENTIRE list is re-sent to the model on every step. That is the memory.
    print("\n" + "=" * 70)
    print("'MEMORY' TRACE (accumulated message list):")
    print("=" * 70)
    for i, m in enumerate(result["messages"]):
        kind = m.__class__.__name__
        if getattr(m, "tool_calls", None):
            calls = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
            body = f"[wants to call] {calls}"
        else:
            body = (m.content or "").replace("\n", " ")[:200]
        print(f"\n[{i}] {kind}: {body}")

    print("\n" + "=" * 70)
    print("FINAL ANSWER:")
    print("=" * 70)
    print(result["messages"][-1].content)
