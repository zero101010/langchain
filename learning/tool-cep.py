from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool

from pydantic import BaseModel, Field, field_validator
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage


@wrap_tool_call
async def error_handler(request, handler):
    try:
        return await handler(request)
    except Exception as e:
        tool_call_id = request.tool_call["id"]
        return ToolMessage(content=f"Erro ao executar a ferramenta: {request.tool_call['name']}. Detalhes do erro: {str(e)}",tool_call_id=tool_call_id)   

class CepInput(BaseModel):
    cep: str = Field(..., description="CEP a ser consultado, deve conter apenas números e ter exatamente 8 dígitos no padrão brasileiro")

    @field_validator('cep')
    @classmethod
    def validate_cep(cls, value):
        cep_lint = value.replace("-", "").replace(".", "")
        if not cep_lint.isdigit() or len(cep_lint) != 8:
            raise ValueError("CEP inválido. Certifique-se de que o CEP contém apenas números e possui 8 dígitos.")
        return value


load_dotenv()



@tool
def find_cep(cep: str) -> str:
    """
    Encontra o endereço a partir do CEP
    Use quando o usuário pedir para encontrar o endereço a partir do CEP
    args:
        cep: CEP a ser consultado, não aceite valores que não sejam números ou que tenham mais ou menos de 8 dígitos e que tenha characteres ou string
    """

    import requests
    # response = requests.get(f"https://viacp.com.br/ws/{cep}/json/")

    response = requests.get(f"https://viacep.com.br/ws/{cep}/json/")
    if response.status_code == 200:
        data = response.json()
        return f"{data['logradouro']}, {data['bairro']}, {data['localidade']}-{data['uf']}"
    else:
        return "Não foi possível encontrar o endereço para o CEP fornecido."


model = init_chat_model("google_genai:gemini-2.5-flash")

agent2 = create_agent(
    model=model,
    system_prompt="Você é um assistente que responde perguntas sobre o que você sabe, o que não souber diga que não sabe a resposta",
    tools=[find_cep],
    middleware=[error_handler]
)
