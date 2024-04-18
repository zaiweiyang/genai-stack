import os
import json
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

load_dotenv(".env")


url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
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

# Helper function to load the upload status
def load_status():
    if os.path.exists(status_file_path):
        with open(status_file_path, 'r') as file:
            return json.load(file)
    else:
        return {}

# Helper function to save the upload status
def save_status(status):
    with open(status_file_path, 'w') as file:
        json.dump(status, file)


def main():
    # st.header("📄Chat with your pdf file")
    
    st.header("📄Demo: Generative AI enriched with local data. \n Locally hosted LLM Model and local Knowledge Base generated from PDFs.")

     # Load upload status
    upload_status = load_status()

    # upload a your pdf file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        if pdf.name in upload_status:
            st.success("This PDF has already been uploaded.")
            metadata = upload_status[pdf.name]
            st.markdown(f"**Title:** {metadata['title']}")
            st.markdown(f"**Abstract:** {metadata['abstract']}")
        else:
            pdf_reader = PdfReader(pdf)

            title = pdf_reader.metadata.get('/Title', "No Title Available") if pdf_reader.metadata else "No Metadata Found"

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            abstract = text[:500] # Use the first 500 characters as an abstract

            # langchain_textspliter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )

            chunks = text_splitter.split_text(text=text)

            # Store the chunks part in db (vector)
            vectorstore = Neo4jVector.from_texts(
                chunks,
                url=url,
                username=username,
                password=password,
                embedding=embeddings,
                index_name="pdf_bot",
                node_label="PdfBotChunk"
                # pre_delete_collection=True,  # Delete existing PDF data
            )
            
            upload_status[pdf.name] = {"title": title, "abstract": abstract}
            save_status(upload_status)

            # qa = RetrievalQA.from_chain_type(
            #     llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
            # )

    upload_status = load_status()
    if upload_status:
        st.success("Previously uploaded PDF: " + ', '.join(upload_status.keys()))
        for pdf_name, info in upload_status.items():
            st.markdown(f"**{pdf_name}** \n - Title: {info['title']} \n - Abstract: {info['abstract'][:100]}...")

        # Use a separate retriever for queries
        retriever = Neo4jVector(
            url=url,
            username=username,
            password=password,
            embedding=embeddings,
            index_name="pdf_bot",
            node_label="PdfBotChunk",
        ).as_retriever()

        # Setup the QA system
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever
        )
        
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file")

        if query:
            stream_handler = StreamHandler(st.empty())
            qa.run(query, callbacks=[stream_handler])


if __name__ == "__main__":
    main()
