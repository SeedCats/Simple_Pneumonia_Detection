"""
AI Agent Module
Handles Grok API integration and medical consultation
"""

import requests
from typing import List, Tuple, Optional, Dict
from config import GROK_API_KEY, GROK_API_URL, GROK_MODEL_NAME, TOP_K_RETRIEVAL


def call_grok_api(messages: List[Dict], rag_retriever=None) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Grok API with website-based RAG enhancement

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        rag_retriever: RAG system instance for knowledge retrieval

    Returns:
        Tuple of (response_content, error_message)
    """
    if not GROK_API_KEY:
        return None, "Grok API Key not set."

    try:
        messages_to_send = [msg.copy() for msg in messages]
        system_message = None  # Initialize

        # Retrieve RAG context if available
        if rag_retriever and messages_to_send:
            system_message = build_system_message_with_rag(messages_to_send, rag_retriever)

        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }

        # Include system message in payload
        final_messages = messages_to_send
        if system_message:
            final_messages = [{"role": "system", "content": system_message}] + messages_to_send

        payload = {
            "model": GROK_MODEL_NAME,
            "messages": final_messages,
            "temperature": 0.5,
            "max_tokens": 5500
        }

        response = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content'], None
        else:
            return None, f"Unexpected API response: {result}"

    except Exception as e:
        return None, f"API request failed: {e}"


def build_system_message_with_rag(messages: List[Dict], rag_retriever) -> Optional[str]:
    """
    Build system message with RAG context

    Args:
        messages: List of messages to extract user query
        rag_retriever: RAG system instance

    Returns:
        System message with RAG context or None
    """
    # Extract last user message
    last_user_message = None
    for msg in reversed(messages):
        if msg['role'] == 'user':
            last_user_message = msg['content']
            break

    if not last_user_message:
        return None

    # Retrieve relevant documents
    retrieved_docs = rag_retriever.retrieve(last_user_message, top_k=TOP_K_RETRIEVAL)
    print(f"✓ RAG RETRIEVAL: Retrieved {len(retrieved_docs)} documents")

    if not retrieved_docs:
        return None

    # Build RAG context
    rag_context = "Available Medical References:\n"
    for i, doc in enumerate(retrieved_docs, 1):
        rag_context += f"{i}. {doc['title']}\n   Source: {doc.get('url', 'N/A')}\n\n"

    system_message = f"""You are a medical AI assistant. 
When answering medical questions, cite relevant sources from the provided medical references.
Use format: [Ref 1], [Ref 2], etc.

Medical Knowledge Base:
{rag_context}

Answer using these references when relevant."""

    return system_message


def generate_initial_analysis_prompt(prediction: str, confidence: float, features: Dict) -> str:
    """
    Generate initial analysis prompt for medical consultation

    Args:
        prediction: Model prediction (Normal/Pneumonia)
        confidence: Confidence score (0-1)
        features: Dictionary of image features

    Returns:
        Formatted prompt string
    """
    initial_prompt = f"""
    As an advanced medical AI diagnostic assistant, analyze this comprehensive pneumonia detection result:

    PRIMARY DIAGNOSIS:
    - Prediction: {prediction}
    - Confidence Score: {confidence:.4f} ({confidence * 100:.1f}%)
    - Image Type: Chest X-ray

    IMAGE CHARACTERISTICS (quantified image features from analysis):
    - Mean Intensity (0-255): {features.get('mean_intensity', 'N/A'):.1f}
    - Intensity Standard Deviation: {features.get('std_intensity', 'N/A'):.1f}
    - Edge Density (0-1): {features.get('edge_density', 'N/A'):.4f}
    - Texture Complexity (Sobel std): {features.get('texture_complexity', 'N/A'):.1f}
    - Histogram Entropy: {features.get('histogram_entropy', 'N/A'):.2f}

    Also, consider that a Grad-CAM heatmap is available, visually indicating the regions of the X-ray image that most strongly influenced the AI's prediction. This heatmap can show areas of high "attention" for the predicted class.

    Please provide a comprehensive analysis including:
    1. Clinical interpretation of the diagnosis, integrating the confidence score.
    2. Analysis of the provided image characteristics in a medical context relevant to pneumonia detection.
    3. How the Grad-CAM heatmap (visualizing AI's focus) might support or challenge the diagnosis.
    4. Potential differential diagnoses to consider based on the findings.
    5. Recommended next steps and clinical validation requirements.
    6. Limitations of this AI analysis and the overall AI assistant role.

    Provide this as a structured medical consultation report AND Please cite the medical references provided above when relevant to your answer.
    """

    return initial_prompt


def format_conversation_for_display(role: str, text: str) -> str:
    """
    Format conversation message for display

    Args:
        role: Message role (user/assistant)
        text: Message text

    Returns:
        Formatted message string
    """
    if role == "user":
        return f"\n\n[You]: {text}"
    elif role == "assistant":
        return f"\n\n[AI Agent]: {text}"
    else:
        return f"\n\n{text}"

