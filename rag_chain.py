import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------
# Load Chroma Vector DB
# ---------------------------------------
def load_vector_db(db_path="chroma_db"):
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(
        name="weather_rag",
        embedding_function=embedding_fn
    )

    return collection


# ---------------------------------------
# RAG Answering Function
# ---------------------------------------
def answer_question(question, groq_api_key):

    collection = load_vector_db()

    # Retrieve Top 5 relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=5
    )

    retrieved_text = "\n\n".join(results["documents"][0])

    # LLM Model
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

    prompt = f"""
    You are an expert weather assistant.
    Use ONLY the following information from the dataset to answer:

    Context:
    {retrieved_text}

    Question:
    {question}

    If the dataset does not contain the answer, say:
    "The dataset does not provide this information."
    """

    response = llm.invoke(prompt)
    return response.content


# ---------------------------------------
# Manual Test
# ---------------------------------------
if __name__ == "__main__":
    key = input("Enter Groq API Key: ")
    q = input("Ask a question: ")

    answer = answer_question(q, key)
    print("\nANSWER:\n", answer)
