import os

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.schema import BaseRetriever

from utils.helpers import LLM_MODEL, DEEPSEEK_BASE_URL


def build_chain(retriever: BaseRetriever) -> ConversationalRetrievalChain:
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model=LLM_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            streaming=True,
            temperature=0,
        ),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
