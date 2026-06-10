from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever


def build_retriever(vectorstore: Chroma) -> BaseRetriever:
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )
