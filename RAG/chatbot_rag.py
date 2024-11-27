import os
from typing_extensions import List, TypedDict

from sentence_transformers import SentenceTransformer

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from pinecone import Pinecone, ServerlessSpec

import streamlit as st

# Configuraciones
index_name = "cvs-ceia"
namespace = "espacio"

# Wrapper para poder usar sentence transformers de huggingface
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        """Embed a list of documents."""
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        """Embed a single query."""
        return self.model.encode([text], convert_to_tensor=False).tolist()

# Creo el objeto para pasar a PineconeVectorStore
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="cpu")
embedding_wrapper = SentenceTransformerEmbeddings(embed_model)


# Conexión a base de datos PineCone
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
pc=Pinecone(api_key=PINECONE_API_KEY)
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = 'cvs-ceia'
# Creación del vectorstore
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding_wrapper,
    namespace=namespace,
)

# Instanciaicón del LLM
llm = ChatGroq(model="llama3-8b-8192")


# Se define un template de prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a knowledge assistant. Based on the context below, provide a concise and accurate answer to the user's query.
    Be brief.
    When applicable, the output should be in items (using "-" to start an item).

    Conversation History:
    {history}
    ---
    Context:
    {context}
    ---
    Question: {question}
    Answer:
    """
)

# Defino tipo de datos de State
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    history: List[str]  


# Defino función de retrieve
def retrieve(state: State):
    retrieved_docs = vectorstore.similarity_search(state["question"],k=2)
    return {"context": retrieved_docs}

# Defino función que genera la respuesta
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # Formateo la historia como un unico string
    history = "\n".join(state["history"])
    
    # Invoco el prompt con contexto e historia previa
    messages = prompt.invoke({
        "question": state["question"],
        "context": docs_content,
        "history": history
    })

    # print(messages)
    response = llm.invoke(messages)
    
    # Se agrega la pregunta y respuesta a la historia previa
    state["history"].append(f"Q: {state['question']} A: {response.content}")
    
    # Ahora ya es posible devolver la respuesta
    return {"answer": response.content}

# Se compila la aplicación usando StateGraph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# Inicializa el historial de conversación en el estado de la sesión
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = {
    "question": "",
    "context": [],
    "answer": "",
    "history": []  # Empty history
}


# Función para generar la respuesta a través de Streamlit
def generate_response(input_text):
    # Genera la respuesta del chatbot utilizando el modelo LLaMA 3 y el historial de la conversación
    # La función generate que llama graph.invoke se encarga automáticamente de:
    # - Hacer un retrieve en la base de datos vectorial.
    # - Hacer un append del historial anterior.
    st.session_state.conversation_history["question"] = input_text
    response = graph.invoke(st.session_state.conversation_history)
    answer = response["answer"]
    return answer

# Configuración de la interfaz de Streamlit
st.title("Ejemplo de RAG usando LLaMa 3 y Pinecone")
st.subheader("¡Hazme una pregunta!")

user_input = st.text_input("Usuario:", "")

if user_input:
    response = generate_response(user_input)
    st.write(f"**Chatbot**: {response}")