import asyncio
import os
from typing import List

from crawl4ai import (
    AdaptiveConfig,
    AdaptiveCrawler,
    AsyncWebCrawler,
    BFSDeepCrawlStrategy,
    CrawlerRunConfig,
)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from config.settings import vectorstore


# 2. Asynchronous Batch Ingestion Engine
async def index_documents_async(documents: List[Document], batch_size: int = 50):
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    async def add_batch(batch: List[Document]):
        try:
            await vectorstore.aadd_documents(batch)
            print(f"✅ Added batch of {len(batch)} chunks")
            return True
        except Exception as e:
            print(f"❌ Error adding batch: {e}")
            return False

    tasks = [add_batch(batch) for batch in batches]
    await asyncio.gather(*tasks, return_exceptions=True)
    print(
        f"🎉 Fully ingested {len(documents)} total chunks across {len(batches)} database batches."
    )


# 3. Main File Processing Loop
async def main():
    # Specify the target directory containing your MDX files
    docs_folder_path = "./langchain-docs"

    if not os.path.exists(docs_folder_path):
        print(f"❌ Error: The directory '{docs_folder_path}' does not exist.")
        return

    print(f"📂 Scanning directory for MDX files: {docs_folder_path}...")

    # 4. Initialize the Loader to recursively find all .mdx targets
    # We use glob="**/*.mdx" to search subfolders. Change to glob="*.mdx" for a flat layout.
    loader = DirectoryLoader(
        docs_folder_path,
        glob="**/*.mdx",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    # Load file data synchronously into memory (LangChain handles raw text conversion)
    raw_documents = loader.load()
    print(f"📄 Found and loaded {len(raw_documents)} raw MDX files.")

    if not raw_documents:
        print("⚠️ No documents discovered. Terminating pipeline.")
        return

    # 5. Clean up Metadata paths
    # DirectoryLoader automatically adds a 'source' key to the metadata pointing to the local filepath
    for doc in raw_documents:
        # Optional: Make the path relative or format it cleanly for references later
        doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "Unknown"))

    # 6. Fragment the text layout using Recursive Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    splitted_docs = text_splitter.split_documents(raw_documents)
    print(
        f"✂️ Split {len(raw_documents)} files into {len(splitted_docs)} structural document chunks."
    )

    # 7. Asynchronously trigger DB pipeline
    print("🚀 Uploading embeddings to PGVector database...")
    await index_documents_async(splitted_docs, batch_size=500)
    print("🏁 Local MDX ingestion pipeline complete!")


if __name__ == "__main__":
    asyncio.run(main())
