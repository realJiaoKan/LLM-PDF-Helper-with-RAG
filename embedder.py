from settings import *
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    def __init__(self, embed_model_name, pdf_file,
                 db_file_path=DEFAULT_DB_FILE_PATH, collection_name=DEFAULT_COLLECTION_NAME):
        self.embed_model_name = embed_model_name
        self.embed_dim = EMBED_MODEL_OPTIONS_DIM[embed_model_name]

        self.pdf_file = PyPDFLoader(pdf_file).load_and_split()

        self.db_file_path = db_file_path
        self.collection_name = collection_name
        pass

    def create_milvus_collection(self):
        milvus_client = MilvusClient(uri=self.db_file_path)

        if milvus_client.has_collection(collection_name=self.collection_name):
            print(
                f"Collection '{self.collection_name}' already exists. Dropping it...")
            milvus_client.drop_collection(collection_name=self.collection_name)

        milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embed_dim,
            metric_type="L2",
            consistency_level="Strong"
        )
        return milvus_client

    def split_pdfs(self):
        print("Splitting texts...")

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(self.pdf_file)
        print(f"Total splitted docs have {len(split_docs)} chunks.")
        return split_docs

    def embed_and_insert_docs(self, milvus_client, split_docs):
        embeddings = HuggingFaceEmbeddings(model_name=self.embed_model_name)

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
            collection_name=self.collection_name,
            data=data_batch
        )
        print(f"Insert done. Total inserted: {len(data_batch)}")

    def embed(self):
        print("-" * 20)
        print("Start embedding...")

        milvus_client = self.create_milvus_collection()

        split_docs = self.split_pdfs()

        self.embed_and_insert_docs(milvus_client, split_docs)

        print("Embedding done!")
        print("-" * 20)


if __name__ == "__main__":
    embedder = Embedder(embed_model_name=DEFAULT_EMBED_MODEL_NAME,
                        pdf_file=DEFAULT_PDF_FILE)
    embedder.embed()
