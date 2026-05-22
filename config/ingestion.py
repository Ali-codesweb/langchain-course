import asyncio
import os
from typing import List

import certifi
from crawl4ai import AdaptiveConfig, AdaptiveCrawler, AsyncWebCrawler
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

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


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    async def add_batch(batch: List[Document]):
        try:
            await vectorstore.aadd_documents(batch)
            print(f"✅ Added batch of {len(batch)} documents")
            return True
        except Exception as e:
            print(f"❌ Error adding batch: {e}")
            return False

    tasks = [add_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"Ingested {len(results)} documents in batches.")


# initialize crawl4ai
async def main():

    config = AdaptiveConfig(
        confidence_threshold=0.90,
        max_pages=1000,
        top_k_links=10,
        min_gain_threshold=0.02,
    )

    async with AsyncWebCrawler() as crawler:
        adaptive = AdaptiveCrawler(crawler, config=config)
        result = await adaptive.digest(
            start_url="https://docs.langchain.com/oss/python/langchain/philosophy",
            query="content on ai agents, tools, messaging, memory, streaming, architecture, and core philosophy",
        )
        adaptive.print_stats()
        all_docs = [
            Document(page_content=page.markdown, metadata={"url": page.url})
            for page in result.knowledge_base
            if page.success and page.markdown  # 👈 Crucial safety check
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "],
        )
        splitted_docs = text_splitter.split_documents(all_docs)

        await index_documents_async(splitted_docs, batch_size=500)
        print("Pipline completed")


if __name__ == "__main__":
    asyncio.run(main())
