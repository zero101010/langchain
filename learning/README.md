# LangChain Agents - Learning Project

This project contains examples of AI agents built with **LangChain**, **LangGraph**, and **LangSmith**, using Google Gemini as the LLM provider.

## Agents

### `agents.py` - Conversational Agent with Memory

An interactive chatbot agent that uses **Tavily** for web search and persists conversation history in a **SQLite** database via LangGraph checkpointing. It also includes a `SummarizationMiddleware` that automatically summarizes the conversation after 6 messages (keeping the last 3), preventing context overflow.

### `tools.py` - Custom Tool Agent (Temperature Converter)

Demonstrates how to create a custom tool using the `@tool` decorator. Includes a Celsius-to-Fahrenheit converter that the agent can invoke when the user asks for temperature conversions.

### `tool-cep.py` - CEP Lookup Agent with Validation and Error Handling

A more advanced agent that looks up Brazilian addresses from a CEP (postal code) using the [ViaCEP API](https://viacep.com.br). Features:

- **Pydantic validation** (`CepInput`) to ensure the CEP has exactly 8 digits.
- **Error handling middleware** (`@wrap_tool_call`) that catches exceptions and returns a friendly error message instead of crashing.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- A Google Gemini API key
- A Tavily API key (for web search in `agents.py`)
- A LangSmith API key (for tracing)

## Setup

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Configure environment variables:**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and fill in your API keys:

   ```
   GEMINI_API_KEY=your-gemini-key
   TAVILY_API_KEY=your-tavily-key
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   LANGSMITH_API_KEY=your-langsmith-key
   LANGSMITH_PROJECT="your-project-name"
   ```

## Running the Agents

### Run the conversational agent locally

```bash
uv run python agents.py
```

This starts an interactive loop where you can ask questions. The agent searches the web via Tavily when needed and remembers the conversation across turns (persisted in `checkpoint.db`).

### Run with LangGraph (Dev Server)

The project includes a `langgraph.json` configuration that exposes the CEP agent as a LangGraph API endpoint.

```bash
uv run langgraph dev
```

This starts the LangGraph development server. You can then interact with the agent via the API or the **LangGraph Studio** UI that opens automatically.

The server configuration (`langgraph.json`) maps:
- **Graph `agent`** -> `tool-cep.py:agent2`

### LangSmith Tracing

LangSmith tracing is enabled automatically when the environment variables are set. Once configured:

1. Run any agent (locally or via LangGraph dev server).
2. Open [LangSmith](https://smith.langchain.com) in your browser.
3. Navigate to your project (defined in `LANGSMITH_PROJECT`) to see traces for every agent invocation, including tool calls, LLM inputs/outputs, and latency metrics.

## Project Structure

```
learning/
├── agents.py          # Conversational agent with web search and memory
├── tools.py           # Custom tool example (temperature converter)
├── tool-cep.py        # CEP lookup agent with validation and error handling
├── langgraph.json     # LangGraph server configuration
├── pyproject.toml     # Project dependencies (managed with uv)
├── .env.example       # Environment variables template
└── checkpoint.db      # SQLite database for conversation persistence
```
