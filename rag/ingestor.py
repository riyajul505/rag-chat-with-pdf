import os
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from utils.helpers import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, CHROMA_PERSIST_DIR


def _extract_pages(pdf_path: str) -> list[Document]:
    docs = []
    filename = os.path.basename(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # also pull any tables as markdown-like text
            for table in page.extract_tables():
                rows = ["\t".join(str(c) if c else "" for c in row) for row in table]
                text += "\n" + "\n".join(rows)
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": i},
                ))
    return docs


def _split(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest(pdf_paths: list[str], persist_dir: str = CHROMA_PERSIST_DIR) -> Chroma:
    all_chunks: list[Document] = []
    for path in pdf_paths:
        pages = _extract_pages(path)
        all_chunks.extend(_split(pages))

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def load_vectorstore(persist_dir: str = CHROMA_PERSIST_DIR) -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
