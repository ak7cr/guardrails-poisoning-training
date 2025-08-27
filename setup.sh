#!/bin/bash
# setup.sh - Quick setup script for the vector guardrail system

echo "🚀 Setting up Vector Guardrail Classification System..."
echo "==============================================="

# Check Python version
python_version=$(python --version 2>&1)
echo "✅ Found Python: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📥 Installing required packages..."
pip install -r requirements.txt

# Test installation
echo "🧪 Testing installation..."
python -c "import torch, transformers, sentence_transformers, faiss; print('✅ All core packages installed successfully!')"

# Test vector system
echo "🔍 Testing vector guardrail system..."
if python -c "from vector_guardrail import VectorGuardrail; print('✅ Vector guardrail system ready!')"; then
    echo "🚀 Setting up models..."
    python setup_models.py
    if [ $? -eq 0 ]; then
        echo "🎉 Setup completed successfully!"
        echo ""
        echo "Next steps:"
        echo "1. Test text classification: python vector_text_test.py \"test message\""
        echo "2. Test document classification: python vector_document_classifier.py --file test/test.csv"
        echo "3. Run benchmarks: python speed_benchmark.py"
        echo ""
        echo "📚 Read README.md for detailed usage instructions"
    else
        echo "❌ Model setup failed. You can still use vector-only mode."
        exit 1
    fi
else
    echo "❌ Setup failed. Please check error messages above."
    exit 1
fi
