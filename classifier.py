import streamlit as st
import torch
from transformers import pipeline

# Global variable fallback
_classifier = None

# Classification Labels
URGENCY_LABELS = ["Urgent", "Normal", "FYI"]
ACTION_LABELS = ["Approve", "Reply", "Provide Info", "Review", "No Action"]

@st.cache_resource(show_spinner=False)
def get_cached_classifier():
    """
    Loads the model ONCE and caches it in memory.
    This prevents the app from 'hanging' on every single click.
    """
    # 1. Check for GPU
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (GeForce GTX)" if device == 0 else "CPU"
    
    print(f"🚀 Initializing AI Classifier on {device_name} (This may take a minute)...")

    # 2. Load Model (DistilBERT - Fast & SafeTensors compatible)
    classifier = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli", 
        device=device
    )
    print("✅ Classifier Loaded & Ready.")
    return classifier

def classify_urgency_and_action(inputs):
    """
    Takes a string (or list of strings) and returns the Urgency and Action tags.
    """
    # Load from cache (Instant after first run)
    classifier = get_cached_classifier()

    # Handle single string input gracefully
    is_single = isinstance(inputs, str)
    texts = [inputs] if is_single else inputs

    # Batch Process
    urgency_results = classifier(texts, URGENCY_LABELS)
    action_results = classifier(texts, ACTION_LABELS)

    # Ensure results are always lists
    if isinstance(urgency_results, dict): urgency_results = [urgency_results]
    if isinstance(action_results, dict): action_results = [action_results]

    final_results = []
    
    # Zip results
    for u, a in zip(urgency_results, action_results):
        top_urgency = u['labels'][0]
        top_action = a['labels'][0]

        formatted_tag = f"{top_urgency} {'❗' if top_urgency == 'Urgent' else ''}".strip()

        final_results.append({
            "urgency": top_urgency,  # Required by backend
            "tag": formatted_tag,    # Required by UI
            "action": top_action
        })

    return final_results[0] if is_single else final_results