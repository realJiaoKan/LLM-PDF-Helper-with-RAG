from settings import *
from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings
import openai

prompt_template = (
    "You are a helpful assistant.\n"
    "Based on the question from the user, I have prepared some context that may be related to the question, "
    "which is given below:\n"
    "{}\n\n"
    "And here is the question: {}\n\n"
    "Please provide a useful answer."
)


def milvus_search(query: str, top_k: int = 3):
    # Initialize Milvus client
    milvus_client = MilvusClient(uri=DB_FILE_PATH)

    # Prepare embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    query_vec = embeddings.embed_query(query)

    # Search in Milvus
    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
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


def generate_answer_with_gpt(query: str, context_list):
    context_str = "\n".join(f"- {c}" for c in context_list)
    prompt = prompt_template.format(context_str, query)

    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    ans = response.choices[0].message.content
    return prompt, ans


def main():
    user_question = input("Enter your question about WinRAR: ")
    print(f"\n[User Question] {user_question}")

    print("------ Searching Milvus... ------")
    top_docs = milvus_search(user_question, top_k=5)
    for i, d in enumerate(top_docs):
        print(f"Doc {i+1} - Score {d['score']} : {d['text']}")

    context_texts = [d["text"] for d in top_docs]

    print("------ Generating answer with GPT... ------")
    prompt, answer = generate_answer_with_gpt(user_question, context_texts)
    print("\n===== GPT Prompt =====")
    print(prompt)
    print("\n===== RAG Answer =====")
    print(answer)


if __name__ == "__main__":
    main()
