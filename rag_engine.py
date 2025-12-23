import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
EMBEDDING_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
VECTOR_DB_PATH = "./inbox_memory_db"
GENERATION_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
TOP_K = int(os.environ.get("TOP_K", 4))

_embedder = None
_client = None
_collection = None

def get_components():
    global _embedder, _client, _collection
    
    if _embedder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        
    if _client is None:
        _client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        _collection = _client.get_or_create_collection(name="emails")
        
    return _embedder, _collection

def index_emails_to_vector_db(emails):
    embedder, collection = get_components()
    ids, documents, metadatas = [], [], []

    try:
        existing_ids = set(collection.get()["ids"])
    except:
        existing_ids = set()

    for email in emails:
        eid = email.get("id")
        if eid in existing_ids: continue

        text = f"Subject: {email.get('subject')}\nFrom: {email.get('sender')}\nBody: {email.get('body')}"
        ids.append(eid)
        documents.append(text)
        metadatas.append({"sender": email.get("sender"), "subject": email.get("subject")})

    if not documents: return 0
    
    embeddings = embedder.encode(documents).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(documents)

def chat_with_inbox(user_query):
    embedder, collection = get_components()
    
    # 1. Retrieve relevant emails
    query_vector = embedder.encode([user_query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=TOP_K)
    
    docs = results['documents'][0] if results['documents'] else []
    if not docs: return "I haven't learned anything from your inbox yet. Hit Sync!"
    
    context = "\n---\n".join(docs)
    
    # 2. Generate Answer using Claude
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key: return "Error: ANTHROPIC_API_KEY not set."
    
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""
You are an intelligent executive assistant. Answer the user's question based ONLY on the emails provided below.

USER QUESTION: {user_query}

EMAILS (CONTEXT):
{context}

ANSWER:
"""
    try:
        message = client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"