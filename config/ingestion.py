import os

import certifi
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# initilize embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")


# initialize pgvector store
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="langchain-docs-2026",
    connection=os.environ.get("PGVECTOR_DB_URL"),
    async_mode=True,
)
