#Importar las librerias requeridas
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Función para extraer el contenido de los archivos PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for num_page, page in enumerate(pdf_reader.pages):
            if num_page == len(pdf_reader.pages) - 1:
                text += '\n\n'
            text += page.extract_text()
    return text

# Función para crear los chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=800,
        chunk_overlap=160,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Función para definir la tienda de vectores
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Función para llamar al modelo de chat
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model='gpt-3.5-turbo-0613')
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Función principal de la app
def main():
    # Configuración de la app
    load_dotenv()
    st.set_page_config(page_title="TalentGenius", page_icon="🧠")
    st.header("Chatea con el genio del reclutamiento 🤖")

    # Inicialización de las sesiones de estado para la conversación
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
    
    # Contenedor de chat
    if user_question := st.chat_input("Has preguntas relacionadas con los archivos de tus candidatos"):
        # Llamado al modelo de chat
        response = st.session_state.conversation({'question': user_question})

        # Asignación de pregunta y respuesta en el chat
        st.session_state.chat_history.extend([
            {"role": "user", "content": response['question']},
            {"role": "assistant", "content": response['answer']}])
        
        # Loop para imprimir los mensajes en el chat
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Barra lateral de la App
    with st.sidebar:
        # Cargue de documentos
        st.header("Tus documentos")
        pdf_docs = st.file_uploader("Carga las hojas de vida de tus candidatos y luego da clic en analizar", accept_multiple_files=True)

        # Condición para evaluar si se subieron documentos
        if pdf_docs:
            button_analyse = st.button("Analizar")
            if button_analyse:
                # Animación de carga y procesamiento de los archivos
                with st.spinner("Analizando"):
                    # Extraer y concatenar texto de los pdf
                    raw_text = get_pdf_text(pdf_docs)
                    # División del texto en chunks
                    text_chunks = get_text_chunks(raw_text)
                    # Creación de la base de datos de vectores
                    vectorstore = get_vectorstore(text_chunks)
                    # Se inicializa la conversación con el modelo de chat
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    # Se notifica al usuario que se han procesado los archivos
                    st.write("¡Archivos analizados exitosamente!, ahora puedes realizar tus consultas")

if __name__ == '__main__':
    main()