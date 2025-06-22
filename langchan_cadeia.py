import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.globals import set_debug

load_dotenv()
# set_debug(True) # pra ver infos sobre a execução entre cadeias

api_key = os.getenv("GEMINI_API_KEY") 

# instancia do modelo
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=api_key)


## modelos prompts
modelo_emocoes = ChatPromptTemplate.from_template(
    "Identifique as emoções presentes na seguinte texto conforme a teoria da roda das emoções. texto: {texto}"
)    

modelo_distorcoes_cognitivas = ChatPromptTemplate.from_template(
    "Identifique as distorções cognitivas a seguinte analise emocional: {emoções}"
)

modelo_pensamentos_automaticos = ChatPromptTemplate.from_template(
    "Sugira pensamentos saudáveis de enfrentamento dadas as seguintes distorções cognitivas: {distorções}. Vc deve montar um esquema tipo cartões de enfrentamento, estilo distorção -> pensamento saudável"
)

## criação das cadeia, uma a uma 

cadeia_emocoes = LLMChain(
    llm=llm,
    prompt=modelo_emocoes
)

cadeia_distorcoes = LLMChain(
    llm=llm,
    prompt=modelo_distorcoes_cognitivas
)

cadeia_pensamentos = LLMChain(
    llm=llm,
    prompt=modelo_pensamentos_automaticos
)


# criação de uma sequência de cadeias -  simple sequential chain
# usa a saída de uma cadeia como entrada da próxima
# a saída da cadeia_emocoes é a entrada da cadeia_distorcoes, e assim por diante

cadeia = SimpleSequentialChain(
    chains= [cadeia_emocoes, cadeia_distorcoes, cadeia_pensamentos],
    verbose=True
)

# execução da cadeia
texto = "E não me sinto bem. Experimente: se você fosse você, como seria e o que faria? Logo de início se sente um constrangimento: a mentira em que nos acomodamos acabou de ser levemente locomovida do lugar onde se acomodara. No entanto já li biografias de pessoas que de repente passavam a ser elas mesmas, e mudavam inteiramente de vida. Acho que se eu fosse realmente eu, os amigos não me cumprimentariam na rua porque até minha fisionomia teria mudado. Como? Não sei."

resultado = cadeia.invoke(texto)

print(resultado)