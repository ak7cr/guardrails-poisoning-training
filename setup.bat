@echo off
REM setup.bat - Quick setup script for# Test vector system
echo 🔍 Testing vector guardrail system...
python -c "from vector_guardrail import VectorGuardrail; print('✅ Vector guardrail system ready!')"
if errorlevel 1 (
    echo ❌ Vector system test failed
    pause
    exit /b 1
)

REM Setup models
echo 🚀 Setting up models...
python setup_models.py
if errorlevel 1 (
    echo ❌ Model setup failed
    pause
    exit /b 1
)

echo 🚀 Setting up Vector Guardrail Classification System...
echo ===============================================

REM Check Python version
python --version
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📥 Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

REM Test installation
echo 🧪 Testing installation...
python -c "import torch, transformers, sentence_transformers, faiss; print('✅ All core packages installed successfully!')"
if errorlevel 1 (
    echo ❌ Package installation verification failed
    pause
    exit /b 1
)

REM Test vector system
echo 🔍 Testing vector guardrail system...
python -c "from vector_guardrail import VectorGuardrail; print('✅ Vector guardrail system ready!')"
if errorlevel 1 (
    echo ❌ Vector system test failed
    pause
    exit /b 1
)

echo 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Test text classification: python vector_text_test.py "test message"
echo 2. Test document classification: python vector_document_classifier.py --file test/test.csv
echo 3. Run benchmarks: python speed_benchmark.py
echo.
echo 📚 Read VECTOR_SYSTEM_README.md for detailed usage instructions
pause
