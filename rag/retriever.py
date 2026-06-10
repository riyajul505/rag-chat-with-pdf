from langchain_community.vectorstores import Chroma
from langchain.schema import BaseRetriever


def build_retriever(vectorstore: Chroma) -> BaseRetriever:
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )
