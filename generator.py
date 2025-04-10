from settings import *
from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings
import openai


class Generator:
    def __init__(self, openai_model, embed_model_name, temerature=0.0,
                 db_file_path=DEFAULT_DB_FILE_PATH, collection_name=DEFAULT_COLLECTION_NAME):
        self.openai_model = openai_model
        self.embed_model_name = embed_model_name
        self.temperature = temerature
        self.db_file_path = db_file_path
        self.collection_name = collection_name
        self.prompt_template = (
            "You are a helpful assistant.\n"
            "Based on the question from the user, I have prepared some context that may be related to the question, "
            "which is given below:\n"
            "{}\n\n"
            "And here is the question: {}\n\n"
            "Please provide a useful and accurate answer."
        )

    def milvus_search(self, query: str, top_k: int = 5):
        # Initialize Milvus client
        milvus_client = MilvusClient(uri=self.db_file_path)

        # Prepare embeddings
        embeddings = HuggingFaceEmbeddings(model_name=self.embed_model_name)
        query_vec = embeddings.embed_query(query)

        # Search in Milvus
        results = milvus_client.search(
            collection_name=self.collection_name,
            data=[query_vec],
            limit=top_k,
            output_fields=["text"]
        )
        hits = results[0]

        docs_found = []
        for h in hits:
            doc_text = h['entity']['text']
            doc_score = h['distance']
            docs_found.append({
                "text": doc_text,
                "score": doc_score
            })

        return docs_found

    def generate_answer_with_gpt(self, query: str, context_list):
        context_str = "\n".join(f"- {c}" for c in context_list)
        prompt = self.prompt_template.format(context_str, query)

        response = openai.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )

        ans = response.choices[0].message.content
        return prompt, ans

    def generate(self, user_query):
        print("-" * 20)
        print("Start generating...")
        print(f"\n[User Question] {user_query}")

        print("Searching Milvus...")
        print("-"*5, "[Top 5 Contexts]", "-"*5)
        top_docs = self.milvus_search(user_query, top_k=5)
        for i, d in enumerate(top_docs):
            print(f"Doc {i+1} - Score {d['score']} : {d['text'][:50]}...")
        context_texts = [d["text"] for d in top_docs]
        print("-"*5, "----------------", "-"*5)

        print("Generating answer with GPT...")
        prompt, answer = self.generate_answer_with_gpt(
            user_query, context_texts)
        print("===== GPT Prompt =====")
        print(prompt)
        print("===== RAG Answer =====")
        print(answer)
        print("======================")

        print("Generating done!")
        print("-" * 20)
        return top_docs, prompt, answer


if __name__ == "__main__":
    generator = Generator(openai_model=DEFAULT_OPENAI_MODEL,
                          embed_model_name=DEFAULT_EMBED_MODEL_NAME)
    generator.generate(input("Enter your question about the PDF content: "))
