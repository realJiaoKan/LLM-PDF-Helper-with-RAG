from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
PDF_FILE = "WinRAR.pdf"
DB_FILE_PATH = "milvus_db.db"
COLLECTION_NAME = "WinRAR"

EMBED_MODEL_NAME = "intfloat/e5-large-v2"
EMBED_DIM = 1024

# --- Create Milvus Client and Collection ---


def create_milvus_collection():
    milvus_client = MilvusClient(uri=DB_FILE_PATH)
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBED_DIM,
        metric_type="L2",
        consistency_level="Strong"
    )
    return milvus_client


def load_and_split_pdfs():
    print("Loading PDF and splitting texts...")
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load_and_split()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Total splitted docs have {len(split_docs)} chunks.")
    return split_docs


def embed_and_insert_docs(milvus_client, split_docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    data_batch = []
    for i, d in enumerate(tqdm(split_docs, desc="Embedding Documents")):
        vector = embeddings.embed_query(d.page_content)
        data_batch.append({
            "id": i,
            "vector": vector,
            "text": d.page_content.replace("\n", " ")
        })

    print("Inserting vectors into Milvus...")
    milvus_client.insert(
        collection_name=COLLECTION_NAME,
        data=data_batch
    )
    print(f"Insert done. Total inserted: {len(data_batch)}")


def main():
    # 1) Create collection
    milvus_client = create_milvus_collection()

    # 2) Load and split
    split_docs = load_and_split_pdfs()

    # 3) Embed and insert
    embed_and_insert_docs(milvus_client, split_docs)

    print("Embedding done!")


if __name__ == "__main__":
    main()
