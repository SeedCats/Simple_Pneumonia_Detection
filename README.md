# Pneumonia Detection AI System

An AI-powered medical diagnostic tool for detecting pneumonia from chest X-ray images using deep learning, RAG, and AI consultation.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Technologies](#technologies)
- [Model Details](#model-details)
- [Limitations](#limitations)

---

## Overview

The **Pneumonia Detection AI System** combines:
- **Deep Learning Classification**: ResNet50-based pneumonia detection
- **Grad-CAM Visualization**: Interpretable AI predictions
- **RAG Knowledge Base**: Medical web resources integration
- **AI Consultation**: Grok API for detailed analysis
- **PDF Reporting**: Automated comprehensive reports

---

## Features

### 🔍 Core Detection
- Binary classification (Normal/Pneumonia)
- ResNet50 model with confidence scoring
- Real-time analysis

### 📊 Image Analysis
- Quantified features (intensity, edge density, texture, entropy)
- Grad-CAM heatmaps showing diagnostic regions

### 🧠 RAG Medical Knowledge
- Web-based retrieval from Mayo Clinic, CDC, MedlinePlus, Cleveland Clinic
- Sentence-transformer embeddings with FAISS indexing
- Context-aware AI responses

### 🤖 AI Consultation
- Grok-4 API integration
- Multi-turn conversations
- Evidence-based differential diagnoses

### 📄 PDF Reports
- Predictions with confidence scores
- Grad-CAM visualization
- Clinical recommendations and disclaimers

### 🎨 GUI
- TKinter desktop application
- Real-time image preview and heatmap display
- Direct PDF export

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Pneumonia Detection System                  │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │    GUI     │───▶│ Preprocess  │───▶│  ResNet50   │       │
│  │ (TKinter)  │    │ & Features  │    │ Classifier  │       │
│  └────────────┘    └─────────────┘    └─────────────┘       │
│                           │                  │              │
│                     ┌─────┴─────┐      ┌─────┴─────┐        │
│                     │  Grad-CAM │      │    RAG    │        │
│                     │  Heatmap  │      │ Retrieval │        │
│                     └───────────┘      └───────────┘        │
│                           │                  │              │
│                           └────────┬─────────┘              │
│                                    ▼                        │
│                          ┌─────────────────┐                │
│                          │   Grok API      │                │
│                          │  Consultation   │                │
│                          └─────────────────┘                │
│                                    │                        │
│                                    ▼                        │
│                          ┌─────────────────┐                │
│                          │  PDF Report     │                │
│                          └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA GPU (recommended) or CPU

### Setup
```bash
# Clone repository
git clone https://github.com/SeedCats/Simple_Pneumonia_Detection.git

# Install dependencies
pip install -r requirements.txt

# Create .env file with API key
echo "XAI_API_KEY=your_grok_api_key" > .env
```

Get Grok API key from [x.ai Console](https://console.x.ai)

Ensure `resnet50_pneumonia_classifier.pth` is in the project directory.

---

## Configuration

Key settings in `config.py`:

```python
# Model
MODEL_PATH = 'resnet50_pneumonia_classifier.pth'
NUM_CLASSES = 2
IMAGE_SIZE = 224
CLASS_NAMES = ['Normal', 'Pneumonia']

# API
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL_NAME = "grok-4-fast-non-reasoning"

# Medical knowledge sources
MEDICAL_URLS = [
    "https://www.mayoclinic.org/diseases-conditions/pneumonia/...",
    "https://medlineplus.gov/pneumonia.html",
    "https://www.cdc.gov/pneumonia/index.html",
    # ...
]
```

---

## Usage

```bash
python run.py
```

### Workflow
1. **Load Image**: Browse and select chest X-ray (JPEG/PNG)
2. **Run Analysis**: Click "Run AI Analysis" for classification + Grad-CAM
3. **View Results**: See prediction, confidence, features, and heatmap
4. **AI Interpretation**: Click "Get AI Interpretation" for detailed analysis
5. **Ask Questions**: Use chat for follow-up queries
6. **Generate Report**: Export comprehensive PDF report

---

## Technologies

| Category | Technologies |
|----------|-------------|
| Deep Learning | PyTorch, TorchVision, ResNet50 |
| NLP/RAG | Sentence-Transformers, FAISS, BeautifulSoup |
| Visualization | Matplotlib, PIL, OpenCV |
| UI/Reporting | TKinter, FPDF2 |
| APIs | Grok API, Requests |

---

## Model Details

### Architecture
- **Base**: ResNet50 (ImageNet pre-trained)
- **Classifier**: Dropout(0.5) → Linear(2048→2)
- **Input**: 224×224 pixels

### Performance
- Accuracy: ~95%
- Sensitivity: ~90-95%
- Specificity: ~85-90%

### Grad-CAM
Highlights regions driving predictions by computing gradients from the final convolutional layer.

---

## Limitations

### Clinical
- No physical examination capability
- Dependent on X-ray image quality
- No access to patient history/symptoms
- ~5-10% error rate possible

### Technical
- Resolution and artifact sensitivity
- Limited to X-ray modality
- Dataset bias potential

### Others
- Interest use only
- Model may not be the best to use
- RAG sources may not be trusted, no reliable knowledge base and keep updating
---