"""
AI Agent Module - Grok API integration and medical consultation
"""

import requests
from typing import List, Tuple, Optional, Dict
from config import GROK_API_KEY, GROK_API_URL, GROK_MODEL_NAME, TOP_K_RETRIEVAL


def call_grok_api(messages: List[Dict], rag_retriever=None) -> Tuple[Optional[str], Optional[str]]:
    """Call Grok API with optional RAG enhancement"""
    if not GROK_API_KEY:
        return None, "Grok API Key not set."

    try:
        messages_to_send = [msg.copy() for msg in messages]
        system_message = build_system_message_with_rag(messages_to_send, rag_retriever) if rag_retriever else None

        final_messages = ([{"role": "system", "content": system_message}] + messages_to_send) if system_message else messages_to_send

        response = requests.post(
            GROK_API_URL,
            headers={"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROK_MODEL_NAME, "messages": final_messages, "temperature": 0.5, "max_tokens": 5500},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        if result.get('choices'):
            return result['choices'][0]['message']['content'], None
        return None, f"Unexpected API response: {result}"
    except Exception as e:
        return None, f"API request failed: {e}"


def build_system_message_with_rag(messages: List[Dict], rag_retriever) -> Optional[str]:
    """Build system message with RAG context"""
    last_user_msg = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), None)
    if not last_user_msg:
        return None

    docs = rag_retriever.retrieve(last_user_msg, top_k=TOP_K_RETRIEVAL)
    print(f"✓ RAG RETRIEVAL: Retrieved {len(docs)} documents")
    if not docs:
        return None

    rag_context = "\n".join(f"{i}. {doc['title']}\n   Source: {doc.get('url', 'N/A')}" for i, doc in enumerate(docs, 1))

    return f"""You are a medical AI assistant. Cite relevant sources using [Ref 1], [Ref 2], etc.

Medical Knowledge Base:
{rag_context}

Answer using these references when relevant."""


def generate_initial_analysis_prompt(prediction: str, confidence: float, features: Dict) -> str:
    """Generate initial analysis prompt for medical consultation"""
    return f"""As an advanced medical AI diagnostic assistant, analyze this pneumonia detection result:

PRIMARY DIAGNOSIS:
- Prediction: {prediction}
- Confidence: {confidence:.4f} ({confidence * 100:.1f}%)
- Image Type: Chest X-ray

IMAGE CHARACTERISTICS:
- Mean Intensity: {features.get('mean_intensity', 0):.1f}
- Std Deviation: {features.get('std_intensity', 0):.1f}
- Edge Density: {features.get('edge_density', 0):.4f}
- Texture Complexity: {features.get('texture_complexity', 0):.1f}
- Histogram Entropy: {features.get('histogram_entropy', 0):.2f}

A Grad-CAM heatmap shows AI focus regions.

Provide analysis including:
1. Clinical interpretation with confidence assessment
2. Image characteristics analysis
3. Grad-CAM interpretation
4. Differential diagnoses
5. Recommended next steps
6. AI limitations

Provide as structured medical consultation. Cite references when relevant."""


def format_conversation_for_display(role: str, text: str) -> str:
    """Format conversation message for display"""
    prefixes = {"user": "\n\n[You]: ", "assistant": "\n\n[AI Agent]: "}
    return f"{prefixes.get(role, '\n\n')}{text}"
