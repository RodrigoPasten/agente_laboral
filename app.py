import streamlit as st
import os
from dotenv import load_dotenv

# Importar todas las demás bibliotecas necesarias como en tu código original

# Cargar las variables de entorno y configurar el agente como lo haces en tu código original
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationBufferMemory
# Para crear agentes personalizados
from langchain.agents import tool
from langchain.agents import load_tools, initialize_agent, AgentType, create_react_agent, AgentExecutor

from langchain.prompts import (
    # Esta clase se usa para crear una plantilla de conversación completa, que puede incluir mensajes del sistema,
    # del AI y del humano.
    ChatPromptTemplate,
    # Una clase básica para crear plantillas de prompts. Permite definir un texto con variables que se pueden
    # rellenar más tarde
    PromptTemplate,
    # Se usa para crear mensajes que representan instrucciones o contexto del sistema en una conversación
    SystemMessagePromptTemplate,
    # Para crear mensajes que representan respuestas o declaraciones de la IA en una conversación.
    AIMessagePromptTemplate,
    # Se utiliza para crear mensajes que representan entradas o preguntas del usuario humano en una conversación.
    HumanMessagePromptTemplate
)
# PDF loader
from langchain.document_loaders import PyPDFLoader

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Leer la API key desde la variable de entorno
api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI()

# Vector database
from langchain_community.vectorstores import SKLearnVectorStore

persist_path1 = "./embbeding_40horas.db"
funcion_embbeding = OpenAIEmbeddings()
vector_store_40h = SKLearnVectorStore(
    embedding=funcion_embbeding,
    persist_path=persist_path1,
    serializer="parquet")

vector_store_40h.persist()

# segundo documento

persist_path2 = "./codigo_trabajo.db"
vector_store_codigo_trabajo = SKLearnVectorStore(
    embedding=funcion_embbeding,
    persist_path=persist_path2,
    serializer='parquet')

vector_store_codigo_trabajo.persist()

compressor = LLMChainExtractor.from_llm(llm)
compressor_retriever_40_horas = ContextualCompressionRetriever(base_compressor=compressor,
                                                               base_retriever=vector_store_40h.as_retriever())
# Compressor codigo del trabajo
compressor_retriever_codigo_trabajo = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store_codigo_trabajo.as_retriever())

# Memoria del chatbot
memory = ConversationBufferMemory(memory_key='chat_memory')


@tool
def consulta_40horas(text: str) -> str:
    """ Retorna respuestas sobre la ley de 40 horas en Chile. Se espera que la entrada sea una cadena de texto y
    retorna una cadena de texto con el resultado mas relevante. Además, debes explicar el resultado obtenido"""
    compressed_doc = compressor_retriever_40_horas.invoke(text)
    resultado = compressed_doc[0].page_content
    return resultado


@tool
def consulta_legislacion_laboral(text: str) -> str:
    """ Retorna respuestas sobre temas relacionados a legislacion laboral en Chile. Se espera que la entrada sea una cadena de texto y
    retorna una cadena de texto con el resultado mas relevante. Si la respuesta con esta herramienta es relevante, no debes usar ninguna herramienta mas
    ni tu propio conocimiento como llm. El resultado debe ser comprensible a cualquier usuario"""
    compressed_doc_ley = compressor_retriever_codigo_trabajo.invoke(text)
    resultado_ley = compressed_doc_ley[0].page_content
    return resultado_ley


# Herramientas desde langchain
tools = load_tools(['wikipedia', 'llm-math', 'serpapi'], llm=llm)

# Concatenación de herramientas
tools = tools + [consulta_40horas, consulta_legislacion_laboral]

agent = initialize_agent(tools=tools,
                         llm=llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         memory=memory,
                         verbose=True)
result = agent.invoke("Que es una licencia medica")

print(result)

# Configuración de la página de Streamlit
st.set_page_config(page_title="Chatbot Laboral", page_icon=":speech_balloon:")

st.title("Chatbot de Legislación Laboral")

# Inicializar el historial de chat en el estado de la sesión si no existe
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes del chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de entrada para el usuario
if prompt := st.chat_input("Escribe tu pregunta aquí"):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Mostrar el mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Obtener la respuesta del agente
        result = agent.invoke(prompt)
        full_response = result['output']

        # Mostrar la respuesta
        message_placeholder.markdown(full_response)

    # Agregar respuesta del asistente al historial
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Botón para limpiar el historial
if st.button("Limpiar conversación"):
    st.session_state.messages = []

