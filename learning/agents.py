import sqlite3

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv

from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware



# Salva o contexto em um banco de dados SQLite
conn = sqlite3.connect('checkpoint.db', check_same_thread=False)
checkpoint_saver = SqliteSaver(conn)
# Salva o contexto em memória
# checkpoint_saver = InMemorySaver()
load_dotenv()


tavily = TavilySearch()

model = init_chat_model("google_genai:gemini-2.5-flash")


agent_igor = create_agent(
    model=model,
    tools=[tavily],
    system_prompt="Você é um assistente que responde perguntas sobre o que você sabe, o que não souber diga que não sabe a resposta",
    checkpointer=checkpoint_saver,
    middleware=[SummarizationMiddleware(model=model, trigger=("messages",6), keep=("messages",3))]
)

config = { "configurable": {"thread_id": "3"} }
print("agente funcionando")


while True:
    question = input("Faça uma pergunta: ")
    answer = agent_igor.invoke({"messages": [{"role": "user", "content": question}]},config=config)
    print(f"Resposta:", answer["messages"][-1].content)