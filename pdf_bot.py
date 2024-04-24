import os
import json
import subprocess
import datetime
import requests

import streamlit as st
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from streamlit.logger import get_logger
from chains import (
    load_embedding_model,
    load_llm,
)

# load api key lib
from dotenv import load_dotenv

# Operation status flag
operation_in_progress = False

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
backup_api_svc_url = os.getenv("BACKUP_API_SVC_URL")

# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)


embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

# Status file path
status_file_path = '/app/data/upload_status.json'

def handle_pdf_upload(pdf, upload_status):
    pdf_reader = PdfReader(pdf)
    title = pdf_reader.metadata.get('/Title', "No Title Available") if pdf_reader.metadata else "No Metadata Found"
    text = "".join((page.extract_text() or "") for page in pdf_reader.pages)
    abstract = text[:500]  # First 500 characters as an abstract

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=text)

    vectorstore = Neo4jVector.from_texts(
        chunks,
        url=url,
        username=username,
        password=password,
        embedding=embeddings,
        index_name="pdf_bot",
        node_label="PdfBotChunk"
    )
    
    upload_status[pdf.name] = {"title": title, "abstract": abstract}
    save_status(upload_status)

def display_uploaded_pdfs(upload_status):
    st.subheader("Previously uploaded PDFs:")
    for pdf_name, info in upload_status.items():
        st.markdown(f"**{pdf_name}**\n - Title: {info['title']}\n - Abstract: {info['abstract'][:150]}...\n")

def run_qa_section(upload_status):
    st.header("PDF Query Assistant")
    # selected_pdf = st.selectbox('Select a PDF to query:', list(upload_status.keys()))
    # if selected_pdf:
        # Set up the QA system
    retriever = Neo4jVector(
        url=url,
        username=username,
        password=password,
        embedding=embeddings,
        index_name="pdf_bot",
        node_label="PdfBotChunk",
    ).as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    
    query = st.text_input("Ask a question:")
    if query:
        response = qa.ask(query)
        st.write(response)

def load_status():
    if os.path.exists(status_file_path):
        with open(status_file_path, 'r') as file:
            return json.load(file)
    return {}

def save_status(status):
    with open(status_file_path, 'w') as file:
        json.dump(status, file)

def get_backup_tags():
    try:
        # Make an HTTP request to the service that can access the backups
        # response = requests.get('http://192.168.31.32:5050/backups')
        response = requests.get(f'{backup_api_svc_url}/backups')
        if response.status_code == 200:
            # Assuming the endpoint returns a JSON array of directory names (tags)
            backup_tags = response.json()
            return backup_tags
        else:
            print(f"Failed to retrieve backup tags: {response.status_code} {response.text}")
            return []
    except Exception as e:
        print(f"Error retrieving backup tags: {e}")
        return []

def manage_backups():
    global operation_in_progress
    st.header("Database Management")
    col1, col2 = st.columns(2)
    
    # Only enable the input and button if no operation is in progress
    with col1:
        tag = st.text_input('Enter a tag for this backup(optional):', key='backup_tag', disabled=operation_in_progress)
        if st.button('Backup Database', key='backup_button', disabled=operation_in_progress):
            if not tag:
                tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            response = backup_database(tag)
            message = response.get('message', 'Backup failed due to an unexpected error.')
            st.text(message)

    with col2:
        backup_tags = get_backup_tags()  
        tag = st.selectbox('Choose a tag to restore from:', backup_tags, key='restore_tag', disabled=operation_in_progress)
        if st.button('Restore Database', key='restore_button', disabled=operation_in_progress):
            response = restore_database(tag)
            message = response.get('message', 'Restore failed due to an unexpected error.')
            st.text(message)

def backup_database(tag=None):
    global operation_in_progress
    operation_in_progress = True
    data = {"tag": tag} if tag else {}
    # response = requests.post('http://192.168.31.32:5050/backup', json=data)
    response = requests.post(f'{backup_api_svc_url}/backup', json=data)

    operation_in_progress = False
    if response.status_code == 200:
        return response.json()
    else:
        return {"message": f"Backup failed with status {response.status_code}: {response.text}"}

def restore_database(tag):
    global operation_in_progress
    operation_in_progress = True
    data = {"tag": tag}
    response = requests.post(f'{backup_api_svc_url}/restore', json=data)
    operation_in_progress = False
    if response.status_code == 200:
        return response.json()
    else:
        return {"message": f"Restore failed with status {response.status_code}: {response.text}"}

def main():
    st.set_page_config(layout="wide")

    st.header("📄Demo: Generative AI enriched with local data. \nLocally hosted LLM Model and local Knowledge Base generated from PDFs.")

    upload_status = load_status()

    col1, col2 = st.columns([3, 1])  # Adjust the ratio as needed

    with col1:
        # Disable the uploader during the backup/restore operations
        if not operation_in_progress:
            pdf = st.file_uploader("Upload your PDF", type="pdf")
            if pdf and pdf.name not in upload_status:
                handle_pdf_upload(pdf, upload_status)

        if upload_status:       
            st.success("Uploaded PDFs: " + ', '.join(upload_status.keys()))

        # Disable the QA section during the backup/restore operations
        if not operation_in_progress:
            run_qa_section(upload_status) 
        
        manage_backups()

    with col2:
        # st.subheader("Previously uploaded PDFs:")
        if upload_status:
            display_uploaded_pdfs(upload_status)

if __name__ == "__main__":
    main()
