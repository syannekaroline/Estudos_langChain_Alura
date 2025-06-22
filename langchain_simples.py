import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()
# --- CORREÇÃO AQUI ---
# 1. Pegue a chave de API do ambiente, como antes.
api_key = os.getenv("GEMINI_API_KEY") 
# ou "GOOGLE_API_KEY" se você usou esse nome no .env

# 2. Verifique se a chave foi carregada corretamente.
if not api_key:
    raise ValueError("Chave de API do Gemini não encontrada. Verifique seu arquivo .env")


frase= "Ai que vontade de voltar pra casa( ainda nem saí kk)"

modelo_do_prompt = PromptTemplate.from_template("Identifique as emoções presentes na seguinte frase conforme a teoria da roda das emoções. frase: {frase}")

prompt = modelo_do_prompt.format(frase=frase)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7,google_api_key=api_key)

resposta= llm.invoke(prompt)

print(resposta.content)