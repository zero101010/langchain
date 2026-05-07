from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool


load_dotenv()

@tool
def convert_to_fahrenheit(celsius: int) -> int:
    """
    Converte a temperatura de Celsius para Fahrenheit
    Use quando o usuário pedir para converter a temperatura de Celsius para Fahrenheit
    args:
        celsius: temperatura em Celsius
    """
    return (celsius * 9/5) + 32


model = init_chat_model("google_genai:gemini-2.5-flash")

agent2 = create_agent(
    model=model,
    system_prompt="Você é um assistente que responde perguntas sobre o que você sabe, o que não souber diga que não sabe a resposta",
    tools=[convert_to_fahrenheit]
)
