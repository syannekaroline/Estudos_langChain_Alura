from dotenv import load_dotenv
import google.genai as genai
import os

# Carrega as variáveis do arquivo .env para o ambiente
load_dotenv()

# 1. Carregue a chave do ambiente para uma variável Python
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Chave de API não encontrada. Verifique se o arquivo .env está correto e no mesmo diretório que o main.py")


frase= "Ai que vontade de voltar pra casa( ainda nem saí kk)"
prompt=f"Identifique as emoções presentes na seguinte frase conforme a teoria da roda das emoções. frase: {frase}"

cliente = genai.Client(api_key=api_key)

response = cliente.models.generate_content(
    model="gemini-1.5-flash",
    contents=prompt
)

print(response.text)