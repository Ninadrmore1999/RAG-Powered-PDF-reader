from models.embeddings import get_embedding
from models.llm import ask_llm

def generate_answer(question, vector_store):
    q_embed = get_embedding(question)
    retrived_chunk = vector_store.search(q_embed, top_k = 4)
    context = "\n\n".join(retrived_chunk)
    return ask_llm(question, context), context