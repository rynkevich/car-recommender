import json
import os.path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter

import config

DATA_PATH = 'data/cars.json'
CHROMA_PATH = 'db'


def load_data() -> list[Document]:
    with open(DATA_PATH, 'r') as file:
        json_data = json.load(file)
        splitter = RecursiveJsonSplitter()
        documents = splitter.create_documents(texts=json_data)
        return documents


def setup_database() -> Chroma:
    if os.path.exists(CHROMA_PATH):
        client_settings = chromadb.config.Settings(is_persistent=True, persist_directory=CHROMA_PATH)
        chroma_client = chromadb.Client(client_settings)
        return Chroma(client=chroma_client, embedding_function=OpenAIEmbeddings())

    documents = load_data()
    return Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)


if __name__ == '__main__':
    config.initialize()
    setup_database()
