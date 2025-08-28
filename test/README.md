# Test Files for Vector Guardrail System

This folder contains sample files for testing the vector-based classification system.
simple_gpu_train.py


## 📁 Test Files

### CSV Files
- **`test.csv`** - Mixed content CSV with malicious patterns (prompt injection, SQL injection)
- **`benign_products.csv`** - Clean product catalog CSV
- **`test_data.csv`** - Additional test data
- **`products.csv`** - Sample product data

### PDF Files  
- **`mal.pdf`** - Sample malicious PDF with suspicious content patterns
- **`Discussion Section for Research Papers.pdf`** - Academic research paper (benign)
- **`38933_NIT.pdf`** - Document sample
- **`resume.pdf`** - Personal resume document

## 🧪 Usage Examples

### Test CSV Classification
```bash
# Test malicious CSV
python vector_document_classifier.py --file test/test.csv

# Test benign CSV  
python vector_document_classifier.py --file test/benign_products.csv
```

### Test PDF Classification
```bash
# Test malicious PDF
python vector_document_classifier.py --file test/mal.pdf

# Test benign PDF
python vector_document_classifier.py --file "test/Discussion Section for Research Papers.pdf"
```

### Expected Results

| File | Expected Classification | Confidence |
|------|------------------------|------------|
| test.csv | MALICIOUS | ~99.6% |
| benign_products.csv | BENIGN | ~99.9% |
| mal.pdf | MALICIOUS | ~91.3% |
| Discussion Section... | BENIGN | ~100% |

## 🔧 Interactive Testing

Run interactive document classifier:
```bash
python vector_document_classifier.py --interactive
```

Then select files from this test folder to classify them interactively.

## ✅ Validation

These test files validate:
- **PDF text extraction** (multiple formats)
- **CSV structured data processing**
- **Malicious pattern detection** (injection attacks, suspicious code)
- **Benign content recognition** (academic papers, business data)
- **Hybrid classification** (vector + transformer fallback)

Perfect for demonstrating the system capabilities to your team!
