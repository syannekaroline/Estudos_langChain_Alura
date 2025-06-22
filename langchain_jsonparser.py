import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
set_debug(True) # pra ver infos sobre a execução entre cadeias

api_key = os.getenv("GEMINI_API_KEY") 

# instancia do modelo
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=api_key)

## criação da classe pro parser
class AnaliseEmocional(BaseModel):
    emoções: list[str] = Field(description="Emoções identificadas no texto")
    justificativa: str = Field(description="Justificativa para as emoções identificadas")


## definição do parser
parser = JsonOutputParser(pydantic_object=AnaliseEmocional)

## modelos prompts
modelo_emocoes = PromptTemplate(
    template ="""Identifique as emoções presentes na seguinte texto conforme a teoria da roda das emoções.
    Texto: {texto}
    {formatacao_de_saida}
    """,
    input_variables=["texto"],
    partial_variables={"formatacao_de_saida": parser.get_format_instructions()}
) 

classificacao_sentimentos = ChatPromptTemplate.from_template(
    "Classifique a potcentagem dos sentimentos(positivo, negativo ou neutro) presentas na lista de emoções. Ex: 'Na lista estão presentes 50% de emoções negativas e 50% de positivas.' Emoções: {emoções}"

)
## criação das cadeia, uma a uma 

cadeia_emocoes = LLMChain(
    llm=llm,
    prompt=modelo_emocoes
)

cadeia_classificacao_sentimentos = LLMChain(
    llm=llm,
    prompt=classificacao_sentimentos
)

# criação de uma sequência de cadeias -  simple sequential chain
# usa a saída de uma cadeia como entrada da próxima
# a saída da cadeia_emocoes é a entrada da cadeia_distorcoes, e assim por diante

cadeia = SimpleSequentialChain(
    chains= [cadeia_emocoes, cadeia_classificacao_sentimentos],
    verbose=True
)

# execução da cadeia
texto = """E não me sinto bem. Experimente: se você fosse você, como seria e o que faria? 
Logo de início se sente um constrangimento: a mentira em que nos acomodamos acabou de ser 
levemente locomovida do lugar onde se acomodara. No entanto já li biografias de pessoas que 
de repente passavam a ser elas mesmas, e mudavam inteiramente de vida. Acho que se eu fosse 
realmente eu, os amigos não me cumprimentariam na rua porque até minha fisionomia teria mudado. Como? Não sei."""

resultado = cadeia.invoke(texto)

print(resultado)