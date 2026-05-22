import os

from dotenv import load_dotenv
from h11._abnf import chunk_size
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
llm = ChatOllama(model="qwen3.5:4b")


def main():
    loader = TextLoader("./resources/blog.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="documents",
        connection=os.environ.get("PGVECTOR_DB_URL"),
    )
    vector_store.add_documents(texts)

    print("Hello Rag")


if __name__ == "__main__":
    main()
