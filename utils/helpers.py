import tiktoken

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "deepseek-chat"          # DeepSeek-V3; swap to "deepseek-reasoner" for R1
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHROMA_PERSIST_DIR = ".chroma"


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    # tiktoken doesn't have a DeepSeek encoding; GPT-4o tokeniser is close enough
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def format_source(doc) -> str:
    meta = doc.metadata
    filename = meta.get("source", "unknown")
    page = meta.get("page", "?")
    return f"**{filename}** — page {page + 1}"
