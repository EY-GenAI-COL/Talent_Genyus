# Importar las librerias requeridas - Hello world
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Función para extraer el contenido de los archivos PDF
def get_pdf_text(pdf_docs):
    text = ""
    n = 1
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for num_page, page in enumerate(pdf_reader.pages):
            if num_page == len(pdf_reader.pages) - 1:
                text += "\n\n"
            text += "Candidato " + str(n) + "\n"
            text += page.extract_text()
        n = n + 1
    return text


# Función para crear los chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n\n", chunk_size=800, chunk_overlap=160, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Función para definir la tienda de vectores
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Función para llamar al modelo de chat
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def new_docs():
    st.session_state.bttn_visible = True
    st.session_state.process_docs = False


def btn_callback():
    st.session_state.bttn_visible = False
    st.session_state.process_docs = True
    st.session_state.clear_btn = True


def clear_chat():
    st.session_state.conversation = []
    st.session_state.chat_history = []


def clear_btn():
    st.session_state.conversation = []
    st.session_state.chat_history = []
    st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)


def enable_chat():
    st.session_state.chat_off = False


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
    if "bttn_visible" not in st.session_state:
        st.session_state.bttn_visible = False
    if "process_docs" not in st.session_state:
        st.session_state.process_docs = False
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = False
    if "chat_off" not in st.session_state:
        st.session_state.chat_off = True
    if "clear_btn" not in st.session_state:
        st.session_state.clear_btn = False
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = []

    # Barra lateral de la App
    with st.sidebar:
        # Cargue de documentos
        st.header("Tus documentos")
        st.session_state.pdf_docs = st.file_uploader(
            "Carga las hojas de vida de tus candidatos y luego da clic en analizar",
            accept_multiple_files=True,
            on_change=new_docs,
        )
        if not (st.session_state.pdf_docs):
            st.session_state.bttn_visible = False
        # Condición para evaluar si se subieron documentos
        if st.session_state.bttn_visible:
            st.button("Analizar", on_click=btn_callback)

        if st.session_state.process_docs:
            # Animación de carga y procesamiento de los archivos
            with st.spinner("Analizando"):
                # Limpia el canal del chat
                clear_chat()
                # Extraer y concatenar texto de los pdf
                raw_text = get_pdf_text(st.session_state.pdf_docs)
                # st.write(raw_text)
                # División del texto en chunks
                text_chunks = get_text_chunks(raw_text)
                # Creación de la base de datos de vectores
                st.session_state.vectorstore = get_vectorstore(text_chunks)
                # Se inicializa la conversación con el modelo de chat
                st.session_state.conversation = get_conversation_chain(
                    st.session_state.vectorstore
                )
                # Se notifica al usuario que se han procesado los archivos
                st.write(
                    "¡Archivos analizados exitosamente!, ahora puedes realizar tus consultas"
                )
                # Evita volver a entrar al loop
                st.session_state.process_docs = False
                # Habilita el chat
                enable_chat()
                # st.write(text_chunks)
        if st.session_state.clear_btn:
            st.button("Limpiar Chat", on_click=clear_btn)

    # Entrada de texto del chat
    chat_input = st.chat_input(
        "Has preguntas relacionadas con los archivos de tus candidatos",
        disabled=st.session_state.chat_off,
    )
    # Contenedor de chat
    if user_question := chat_input:
        # Llamado al modelo de chat
        response = st.session_state.conversation({"question": user_question})

        # Asignación de pregunta y respuesta en el chat
        st.session_state.chat_history.extend(
            [
                {"role": "user", "content": response["question"]},
                {"role": "assistant", "content": response["answer"]},
            ]
        )

        # Loop para imprimir los mensajes en el chat
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


if __name__ == "__main__":
    main()
