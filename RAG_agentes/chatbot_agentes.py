import os
import re
from typing_extensions import List, TypedDict

from sentence_transformers import SentenceTransformer

from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from pinecone import Pinecone, ServerlessSpec

import streamlit as st

# Configuraciones
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

# Instanciaicón del LLM
llm = ChatGroq(model="llama3-8b-8192")

# Defino tipo de datos de State para usar junto con LangGraph
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    individual: str
    history: List[str]  

# Defino template del prompt
prompt = PromptTemplate(
            input_variables=["context", "question","individual"],
            template="""You are a knowledge assistant that give answer about information inside resumes.
Resume context of individuals will be given to you after "Context:" keyword. Prioritize the latest context rather than your knowledge.
Be brief with your answers, don't exceed more than 200 words, except if you need to finish a sentence.
When applicable, the output should be in items (using "-" to start an item).
If individual data is missing, you must begin the sentence with: "Your question is not related with any of the individuals in my database, valid individuals are 'leandro saraco' or 'elon musk'".

            Conversation History:
            '''{history}'''
            ---
            Context:
            {context}
            ---
            Individual:
            {individual}
            ---
            Question: {question}
            Answer:
            """
)

# Defino el agente. Cada agente levantará datos de la base vectorial, usando un índice por individuo
class Agent:
    """Clase del agente. Hace retrieve de la base vectorial según la persona que se consulte."""
    def __init__(self, embedding_wrapper, index=""):
        if index=="":
            raise ValueError("No se especifico un índice válido.")
        
        self.index = index
        self.embedding_wrapper = embedding_wrapper

        self.vectorstore = PineconeVectorStore(
            index_name=index,
            embedding=self.embedding_wrapper,
            namespace=namespace,
        )

    def get_context(self,state: State):
        retrieved_docs = self.vectorstore.similarity_search(state["question"],k=2)
        return {"context": retrieved_docs}

# Instancio los agentes
leandro_agent = Agent(embedding_wrapper,"leandro-saraco")
elon_agent = Agent(embedding_wrapper,"elon-musk")

# Nodo generate
def generate(state: State):
    if state["context"]:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    else: 
        docs_content = ""
    # Formateo la historia como un unico string
    history = "\n".join(state["history"])
    
    # Invoco el prompt con contexto e historia previa
    messages = prompt.invoke({
        "question": state["question"],
        "context": docs_content,
        "individual": state["individual"],
        "history": history
    })

    # print(messages)
    response = llm.invoke(messages)
    
    state["history"].append(f"Q: {state['question']} A: {response.content}")
    
    # Ahora ya es posible devolver la respuesta
    return {"answer": response.content}

# Nodo para limpiar el contexto
def empty_context(state:State):
    """Limpia el estado del contexto si la pregunta no habla de ningún individuo válido."""
    return {"context":[]}

# Nodo que toma la decisión de contexto segun de quién habla la pregunta
def decide(state: State):
    """Toma la decisión de elegir un agente en base a qué persona se está hablando."""
    leandro_pattern = r"(Leandro\sSaraco|Leandro|Saraco)"
    elon_pattern = r"(Elon\sMusk|Elon|Musk)"
    individual = "" #Default
    if re.search(leandro_pattern, state["question"], re.IGNORECASE):
        individual = "leandro"
    elif re.search(elon_pattern, state["question"], re.IGNORECASE):
        individual = "elon"
    return {"individual":individual}

# Funcion para determinar cuál es el próximo nodo
def decision_read_state(state:State):
    """Obtiene el individuo desde el state y lo retorna para decidir por qué nodo continuar."""
    indiv = state["individual"]
    if indiv=="":
        print("La pregunta no habla de ningun individuo.")
        return "no_individual"
    print("La pregunta habla sobre el individuo:",indiv)
    return indiv



# Armo del grafo usando LangGraph
graph_builder = StateGraph(State)
graph_builder.add_node("decision",decide)
graph_builder.add_node("empty_context",empty_context)
graph_builder.add_node("context_leandro",leandro_agent.get_context)
graph_builder.add_node("context_elon",elon_agent.get_context)
graph_builder.add_node("generate",generate)
graph_builder.add_conditional_edges(
    "decision",
    decision_read_state,
    {"leandro": "context_leandro","elon": "context_elon","no_individual":"empty_context"}
    )
graph_builder.add_edge("context_leandro","generate")
graph_builder.add_edge("context_elon","generate")
graph_builder.add_edge("empty_context","generate")
graph_builder.set_entry_point("decision")
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
st.title("RAG con agentes.")
st.subheader("¡Hazme una pregunta!")

user_input = st.text_input("Usuario:", "")

if user_input:
    response = generate_response(user_input)
    st.write(f"**Chatbot**: {response}")