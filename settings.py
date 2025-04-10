import os
import openai

DEFAULT_PDF_FILE = "WinRAR.pdf"
DEFAULT_DB_FILE_PATH = "milvus_db.db"
DEFAULT_COLLECTION_NAME = "embedded_docs"

DEFAULT_EMBED_MODEL_NAME = "intfloat/e5-large-v2"
EMBED_MODEL_OPTIONS = [
    "intfloat/e5-large-v2",
    "sentence-transformers/all-MiniLM-L6-v2"
]
EMBED_MODEL_OPTIONS_DIM = {
    "intfloat/e5-large-v2": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384
}

DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MODEL_OPTIONS = [
    "gpt-3.5-turbo",
    "gpt-4o"
]
openai.api_key = os.environ["OPENAI_API_KEY"]
