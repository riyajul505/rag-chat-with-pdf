import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

from utils.helpers import LLM_MODEL, DEEPSEEK_BASE_URL

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and a follow-up question, rewrite the follow-up "
     "as a standalone question that can be understood without the history. "
     "Return only the rewritten question."),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering questions about the user's documents. "
     "Use ONLY the context below to answer. If the answer is not in the context, "
     "say you don't know.\n\nContext:\n{context}"),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

HISTORY_WINDOW = 5  # keep the last 5 Q/A pairs


class RAGChain:
    """Conversational retrieval built on langchain-core primitives only.

    Avoids the legacy langchain.chains / langchain.memory modules, which are
    removed in LangChain 1.x and broken on Python 3.14 in 0.3.x.
    """

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
        self.history: list[tuple[str, str]] = []
        llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            temperature=0,
        )
        self._condense = CONDENSE_PROMPT | llm | StrOutputParser()
        self._answer = ANSWER_PROMPT | llm | StrOutputParser()

    def _messages(self) -> list[tuple[str, str]]:
        msgs = []
        for human, ai in self.history[-HISTORY_WINDOW:]:
            msgs.append(("human", human))
            msgs.append(("ai", ai))
        return msgs

    def invoke(self, inputs: dict) -> dict:
        question = inputs["question"]

        # condense follow-ups into a standalone query for retrieval
        search_query = question
        if self.history:
            search_query = self._condense.invoke(
                {"chat_history": self._messages(), "question": question}
            )

        docs = self.retriever.invoke(search_query)
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        answer = self._answer.invoke({
            "context": context,
            "chat_history": self._messages(),
            "question": question,
        })

        self.history.append((question, answer))
        return {"answer": answer, "source_documents": docs}

    def clear_history(self) -> None:
        self.history.clear()


def build_chain(retriever: BaseRetriever) -> RAGChain:
    return RAGChain(retriever)
