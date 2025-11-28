"""
Configuration file for Pneumonia Detection AI System
Centralizes all configuration parameters and constants
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_PATH = 'resnet50_pneumonia_classifier.pth'
NUM_CLASSES = 2
IMAGE_SIZE = 224
CLASS_NAMES = ['Normal', 'Pneumonia']

# ImageNet normalization standard
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ============================================================================
# GROK API CONFIGURATION
# ============================================================================
GROK_API_KEY = os.getenv("XAI_API_KEY")
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL_NAME = "grok-4-fast-non-reasoning"

# ============================================================================
# RAG CONFIGURATION
# ============================================================================
# Medical knowledge base file paths
KB_PATH = "web_medical_kb.pkl"
INDEX_PATH = "web_faiss_index.bin"
TEMP_GRAD_CAM_PATH = "temp_grad_cam_overlay.png"
TEMP_IMAGE_PATH = "temp_original_image.png"

# Medical URLs for RAG knowledge base
MEDICAL_URLS = [
    "https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204",
    "https://medlineplus.gov/pneumonia.html",
    "https://www.cdc.gov/pneumonia/index.html",
    "https://my.clevelandclinic.org/health/diseases/4471-pneumonia"
]

# RAG settings
MAX_PAGES_TO_CRAWL = 20
REQUEST_TIMEOUT = 10
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# ============================================================================
# GUI CONFIGURATION
# ============================================================================
WINDOW_TITLE = "Advanced Pneumonia X-ray Detector with AI Agent"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900

# Font configurations
DEFAULT_FONT_FAMILY = "Arial"
DEFAULT_FONT_SIZE = 10

# ============================================================================
# GRAD-CAM CONFIGURATION
# ============================================================================
GRAD_CAM_CMAP = 'jet'
GRAD_CAM_ALPHA = 0.5

# ============================================================================
# PDF REPORT CONFIGURATION
# ============================================================================
PDF_FONT_DIR = "fonts"
PDF_DEFAULT_FONT = "helvetica"
PDF_ENCODING = "utf-8"
PDF_LINE_HEIGHT = 5

# ============================================================================
# EMBEDDING & RETRIEVAL CONFIGURATION
# ============================================================================
TOP_K_RETRIEVAL = 5 # Get first x most related documents in RAG
EMBEDDING_DIMENSION_FALLBACK = 100

# ============================================================================
# TEXT PROCESSING CONFIGURATION
# ============================================================================
TEXT_REPLACEMENTS = {
    "—": "-",      # em dash to regular dash
    "–": "-",      # en dash to regular dash
    "'": "'",      # left single quotation mark
    "'": "'",      # right single quotation mark
    """: '"',      # left double quotation mark
    """: '"',      # right double quotation mark
    "…": "...",    # ellipsis
    "•": "*",      # bullet point
    "°": " deg",   # degree symbol
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
DEBUG_MODE = False
LOG_LEVEL = "INFO"

# ============================================================================
# VALIDATION & CONSTRAINTS
# ============================================================================
MIN_CONTENT_LENGTH = 50
MAX_CONTENT_LENGTH = 1000
MIN_CONFIDENCE_THRESHOLD = 0.0
MAX_CONFIDENCE_THRESHOLD = 1.0

# ============================================================================
# FILE EXTENSIONS
# ============================================================================
SUPPORTED_IMAGE_FORMATS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
SUPPORTED_IMAGE_TYPES = ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
# Will be set to "cuda:0" if CUDA available, otherwise "cpu"
DEVICE = None  # Dynamically set in application

# ============================================================================
# FALLBACK MEDICAL KNOWLEDGE BASE
# ============================================================================
FALLBACK_MEDICAL_KNOWLEDGE = [
    {
        "id": 1,
        "title": "Pneumonia Diagnostic Standards",
        "content": "Pneumonia is lung inflammation caused by pathogens. Main symptoms include cough, fever, and respiratory difficulty. Chest X-ray examination is key for diagnosis.",
        "url": "fallback"
    },
    {
        "id": 2,
        "title": "Normal Chest X-ray Features",
        "content": "Normal chest X-rays show clear lung fields, normal heart size, and no abnormal shadows. Ribs are symmetrically distributed.",
        "url": "fallback"
    },
    {
        "id": 3,
        "title": "Pneumonia X-ray Manifestations",
        "content": "Pneumonia patients' X-rays show lung infiltration, consolidation shadows, and possible bronchial air signs. Commonly found in lower lobes.",
        "url": "fallback"
    }
]

# ============================================================================
# DISCLAIMER TEXT
# ============================================================================
PDF_DISCLAIMER = (
    "Disclaimer: This report is generated by an AI model and is for informational purposes only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always seek the advice of a qualified healthcare provider for any medical questions or conditions. "
    "The AI model's output should be validated by a human expert."
)

# ============================================================================
# FUNCTION UTILITIES
# ============================================================================

def get_supported_image_filter():
    """Get file dialog filter for supported image formats"""
    return [
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
        ("All files", "*.*")
    ]

def validate_config():
    """Validate critical configuration settings"""
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}")
        return False

    if not GROK_API_KEY:
        print("Warning: GROK_API_KEY not set in environment variables")
        return False

    return True

def print_config_summary():
    """Print current configuration for debugging"""
    print("\n" + "="*60)
    print("PNEUMONIA DETECTION SYSTEM - CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"RAG Knowledge Base Path: {KB_PATH}")
    print(f"Medical URLs: {len(MEDICAL_URLS)} sources")
    print(f"Grok API URL: {GROK_API_URL}")
    print(f"Grok Model: {GROK_MODEL_NAME}")
    print("="*60 + "\n")

