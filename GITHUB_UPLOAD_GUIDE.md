# Essential Files for GitHub Upload

## 🚀 Core System Files (Required)

### Main Classification Engine
- `vector_guardrail.py` - Core vector-based classification system
- `vector_document_classifier.py` - Document classification (PDF/CSV)
- `vector_text_test.py` - Quick text testing tool

### Training & Testing Tools
- `advanced_focal_train.py` - Advanced model training with Focal Loss
- `test_advanced_model.py` - Comprehensive testing framework
- `speed_benchmark.py` - Performance comparison benchmarks

### Vector Database (Essential - Pre-built patterns)
- `vector_database/` folder containing:
  - `malicious_index.faiss` - FAISS index for malicious patterns
  - `benign_index.faiss` - FAISS index for benign patterns  
  - `malicious_vectors.npy` - Malicious pattern embeddings
  - `benign_vectors.npy` - Benign pattern embeddings
  - `malicious_texts.json` - Malicious pattern texts
  - `benign_texts.json` - Benign pattern texts
  - `metadata.json` - Database metadata

### Configuration & Dependencies
- `requirements.txt` - Python package dependencies
- `VECTOR_SYSTEM_README.md` - Complete system documentation
- `.gitignore` - Git ignore rules

## 📦 Optional Files (Team Convenience)

### Sample Test Files
- `test.csv` - Sample CSV with mixed content for testing
- `benign_products.csv` - Sample benign CSV file

### Legacy/Alternative Tools
- `production_guardrail.py` - Alternative production implementation
- `quick_text_test.py` - Alternative text testing tool

## ❌ Files NOT to Upload

### Environment & Local Files
- `venv/` - Virtual environment (team creates their own)
- `__pycache__/` - Python cache files
- `*.pyc` - Compiled Python files

### Large Model Files (Optional - Can Be Downloaded)
- `text_guardrail_advanced_model/` - Trained transformer model (~250MB)
  - Teams can retrain or download pre-trained models
  - Alternative: Upload to GitHub LFS or cloud storage

### Personal/Test Documents
- `38933_NIT.pdf` - Personal document
- `Discussion Section for Research Papers.pdf` - Sample document
- `mal.pdf` - Test document (keep for testing)
- `resume.pdf` - Personal document

### Dataset/Training Folders
- `prompt_injection_dataset/` - Raw training data (large)
- `text_guardrail_advanced/` - Training artifacts
- `modernbert-prompt-detector/` - Alternative model
- `simple-gpu-detector/` - Alternative implementation

## 📋 Complete Upload Checklist

### ✅ Must Upload (Core System)
```
vector_guardrail.py
vector_document_classifier.py  
vector_text_test.py
advanced_focal_train.py
test_advanced_model.py
speed_benchmark.py
vector_database/
  ├── malicious_index.faiss
  ├── benign_index.faiss
  ├── malicious_vectors.npy
  ├── benign_vectors.npy
  ├── malicious_texts.json
  ├── benign_texts.json
  └── metadata.json
test/
  ├── test.csv
  ├── benign_products.csv
  ├── mal.pdf
  └── *.pdf (sample documents)
requirements.txt
VECTOR_SYSTEM_README.md
.gitignore
setup.sh
setup.bat
```

### 🔄 Team Setup Instructions

After cloning, team members need to:

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the system:**
   ```bash
   python vector_text_test.py "test message"
   python vector_document_classifier.py --file test.csv
   ```

4. **Optional: Retrain models:**
   ```bash
   python advanced_focal_train.py  # Trains new transformer model
   ```

## 💡 GitHub Repository Structure

```
guardrail-vector-system/
├── README.md (rename VECTOR_SYSTEM_README.md)
├── requirements.txt
├── .gitignore
├── src/
│   ├── vector_guardrail.py
│   ├── vector_document_classifier.py
│   ├── vector_text_test.py
│   ├── advanced_focal_train.py
│   ├── test_advanced_model.py
│   └── speed_benchmark.py
├── vector_database/
│   ├── *.faiss
│   ├── *.npy
│   └── *.json
├── test/
│   ├── test.csv                     # Sample malicious CSV
│   ├── benign_products.csv          # Sample benign CSV
│   ├── mal.pdf                      # Sample malicious PDF
│   └── *.pdf                        # Other test documents
└── docs/
    └── API_DOCUMENTATION.md
```

## 🔑 Key Benefits for Team

✅ **Instant Setup** - Pre-built vector database ready to use
✅ **No Training Required** - Skip expensive model training
✅ **Multiple Tools** - Text testing, document classification, benchmarking
✅ **Complete Documentation** - Full usage examples and API docs
✅ **Production Ready** - Tested, optimized, and battle-tested
✅ **Cross-Platform** - Works on Windows, Linux, MacOS
✅ **GPU Optional** - Runs efficiently on CPU for vector operations
