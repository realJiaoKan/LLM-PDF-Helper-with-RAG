import os
import openai

PDF_FILE = "WinRAR.pdf"
DB_FILE_PATH = "milvus_db.db"
COLLECTION_NAME = "WinRAR"

EMBED_MODEL_NAME = "intfloat/e5-large-v2"
EMBED_DIM = 1024

OPENAI_MODEL = "gpt-3.5-turbo"
openai.api_key = os.environ["OPENAI_API_KEY"]
