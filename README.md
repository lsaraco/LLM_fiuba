# LLM_fiuba

## TP 1 - RAG

- Link: [https://github.com/lsaraco/LLM_fiuba/tree/main/RAG](https://github.com/lsaraco/LLM_fiuba/tree/main/RAG)
- Implementación de RAG usando LLM y base de datos vectorial.
- Se usa mi CV como fuente de datos para el agregado de contexto.
- Base de datos: Pinecone.
- LLM: LLaMa 3 (8B) a través de Groq.
- Se usa langchain para gestionar retrieval e inferencia.
- Streamlit para generar una aplicación de chat a modo demostrativo.

## TP 2 - RAG con Agentes

- Link: [https://github.com/lsaraco/LLM_fiuba/tree/main/RAG_agentes](https://github.com/lsaraco/LLM_fiuba/tree/main/RAG_agentes)
- Extensión del TP1 utilizando agentes.
- Existen dos agentes con conocimiento específico sobre los CVs de:
    - Leandro Saraco
    - Elon Musk
- Se usa LangGraph con un nodo de decisión para determinar sobre qué individuo se requieren datos.

## TP 3 - Generación de imágenes a partir de texto

- Link: [https://github.com/lsaraco/LLM_fiuba/tree/main/Generacion_imagenes](https://github.com/lsaraco/LLM_fiuba/tree/main/Generacion_imagenes)
- Generación de imágenes a partir de texto.
- Uso de StableDifussion v2 (versión normal y base).
- Uso de librerías de HugginFace.
