from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from transformers import pipeline

# Knowledge Base
knowledge_base = [
    "If you're stressed, try deep breathing: Inhale for 4 seconds, hold for 4, exhale for 4. Repeat 3 times.",
    "Feeling anxious? Ground yourself by naming 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
    "In a crisis? Call 988 (US) or your local hotline for immediate help.",
    "For better sleep, avoid screens 30 minutes before bed and try a calming routine.",
    "If you're overwhelmed, break tasks into small steps and tackle one at a time.",
    "Feeling sad? It’s okay to let yourself feel it—try journaling or talking to a friend.",
    "Hello , Now i am here what i do to help you."
]

def setup_rag():
    # Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Convert knowledge base to documents
    docs = [Document(page_content=text) for text in knowledge_base]
    
    # Create vector store
    vector_store = FAISS.from_documents(docs, embedding_model)
    
    # Generative Model
    generator = pipeline("text-generation", model="distilgpt2", max_length=100)
    
    return vector_store, generator

def mental_health_chatbot(user_input, vector_store, generator):
    # Retrieve relevant info
    retrieved_docs = vector_store.similarity_search(user_input, k=1)
    context = retrieved_docs[0].page_content if retrieved_docs else "I’m here to support you."

    # Craft prompt
    prompt = f"You are a kind, empathetic mental health chatbot. The user says: '{user_input}'. Using this info: '{context}', respond supportively."
    
    # Generate response
    response = generator(prompt, num_return_sequences=1, temperature=0.7)[0]["generated_text"]
    
    # Clean up: Remove prompt from output
    response = response.split("respond supportively.")[-1].strip() if "respond supportively" in response else response
    
    
    crisis_keywords = ["crisis", "hurt myself", "end it", "don’t want to live"]
    if any(keyword in user_input.lower() for keyword in crisis_keywords):
        response += "\nI’m really worried about you. Please call 988 (US) or your local hotline for help right now."
    elif len(response) < 20:  # Fallback for short responses
        response = f"I hear you—{context} Does that sound like something you’d like to try?"

    return response