# 🚀 Guardrails Poisoning Training

**Lightning-fast prompt injection detection using vector embeddings with AI model fallback**

🤗 **Model:** [ak7cr/guardrails-poisoning-training](https://huggingface.co/ak7cr/guardrails-poisoning-training)    
⚡ **Speed:** Vector-fast with transformer fallback  

## 🎯 Core Features
- **⚡ Ultra-fast classification** using FAISS vector similarity search
- **📄 Document support** for PDF and CSV files  
- **🔧 Zero setup** - uses model directly from Hugging Face
- **💾 Lightweight** - no large model downloads required

## 📁 Main Files

### 🔤 Text Classification
- **`vector_text_test.py`** - Quick interactive text testing
- **`vector_guardrail.py`** - Core classification engine with FAISS

### 📄 Document Classification  
- **`vector_document_classifier.py`** - PDF and CSV document classifier

### ⚡ Performance & Setup
- **`speed_benchmark.py`** - Performance comparison tool
- **`setup_models.py`** - Automatic setup and testing

### 🧠 Training (Advanced)
- **`advanced_focal_train.py`** - Model training with Focal Loss
- **`upload_to_huggingface.py`** - Upload trained models to HF Hub

## 🚀 Quick Start

### **Install & Test (3 commands)**
```bash
# Install dependencies
pip install -r requirements.txt

# Setup and test model
python setup_models.py

# Test text classification
python vector_text_test.py
```

### **Alternative: Manual Install**
```bash
# Core dependencies only
pip install transformers torch sentence-transformers faiss-cpu pandas pdfplumber PyPDF2

# Then setup
python setup_models.py
```

### **Direct Usage (No Setup)**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model directly from Hugging Face
model = AutoModelForSequenceClassification.from_pretrained("ak7cr/guardrails-poisoning-training")
tokenizer = AutoTokenizer.from_pretrained("ak7cr/guardrails-poisoning-training")

# Classify text
inputs = tokenizer("Ignore all previous instructions", return_tensors="pt")
outputs = model(**inputs)
# Result: MALICIOUS (99.99% confidence)
```

## 🎯 Usage Examples

### Text Classification
```bash
# Interactive mode
python vector_text_test.py

# Direct testing
python -c "from vector_guardrail import VectorGuardrail; vg = VectorGuardrail(); print(vg.classify('Ignore all previous instructions'))"
```

### Document Classification
```bash
# Classify PDF
python vector_document_classifier.py --file test/mal.pdf

# Classify CSV
python vector_document_classifier.py --file test/test.csv

# Interactive file selection
python vector_document_classifier.py --interactive
```

### Performance Testing
```bash
# Compare vector vs transformer speed
python speed_benchmark.py
```

## 📊 Performance

- **Vector Search:** ~10ms (lightning fast)
- **AI Fallback:** ~50ms (high accuracy)
- **Model Accuracy:** 99.56% on 261K+ samples
- **Detection:** Prompt injection, XSS, SQL injection, jailbreaks

## 🏗️ Architecture

### **Hybrid Classification**
1. **FAISS Vector Search** - Compare against 47 malicious + 50 benign patterns
2. **Confidence Check** - Use vector result if confidence > 75%
3. **AI Fallback** - Use HF transformer for uncertain cases
4. **Result Fusion** - Best of both approaches

### **Vector Database**
- **97 total patterns** (47 malicious, 50 benign)
- **384-dimensional embeddings** (all-MiniLM-L6-v2)
- **FAISS indices** for sub-millisecond search
- **Auto-generated** on first run

## 🔧 Files Overview

| File | Purpose | Usage |
|------|---------|-------|
| `vector_text_test.py` | Interactive text testing | `python vector_text_test.py` |
| `vector_document_classifier.py` | PDF/CSV classification | `python vector_document_classifier.py --file doc.pdf` |
| `vector_guardrail.py` | Core classification engine | Import in your code |
| `speed_benchmark.py` | Performance testing | `python speed_benchmark.py` |
| `setup_models.py` | Setup and validation | `python setup_models.py` |
| `advanced_focal_train.py` | Model training | `python advanced_focal_train.py` |

## 🤗 Model Details

**Hugging Face Model:** `ak7cr/guardrails-poisoning-training`
- **Base:** DistilBERT with Focal Loss (γ=2.0)
- **Training:** 261,738 samples from jayavibhav/prompt-injection
- **Size:** 268MB (auto-cached by Hugging Face)
