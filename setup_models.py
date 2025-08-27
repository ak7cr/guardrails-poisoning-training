#!/usr/bin/env python3
"""
Model Setup Script for Vector Guardrail System
Downloads or trains the required models for the system to work
"""

import os
import sys
import subprocess
from pathlib import Path

def check_model_exists(model_path):
    """Check if a model directory exists and has files"""
    if not os.path.exists(model_path):
        return False
    
    # Check if directory has model files
    model_files = list(Path(model_path).glob("*.bin")) + list(Path(model_path).glob("*.safetensors"))
    return len(model_files) > 0

def setup_vector_database():
    """Set up vector database"""
    print("🔧 Setting up vector database...")
    
    if os.path.exists("vector_database") and len(os.listdir("vector_database")) > 0:
        print("✅ Vector database already exists")
        return True
    
    try:
        # Run vector guardrail to build database
        result = subprocess.run([sys.executable, "vector_guardrail.py"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Vector database created successfully")
            return True
        else:
            print(f"❌ Failed to create vector database: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        return False

def train_advanced_model():
    """Train the advanced transformer model"""
    print("🚀 Training advanced transformer model...")
    print("⚠️  This will take 10-20 minutes with GPU, longer with CPU")
    
    try:
        result = subprocess.run([sys.executable, "advanced_focal_train.py"], 
                              timeout=1800)  # 30 min timeout
        if result.returncode == 0:
            print("✅ Advanced model trained successfully")
            return True
        else:
            print("❌ Model training failed")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Model training timed out")
        return False
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def main():
    print("🚀 Model Setup for Vector Guardrail System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("vector_guardrail.py"):
        print("❌ Error: Please run this script from the main project directory")
        sys.exit(1)
    
    # Setup vector database (always needed, lightweight)
    if not setup_vector_database():
        print("❌ Failed to setup vector database")
        sys.exit(1)
    
    # Check if advanced model exists
    advanced_model_path = "text_guardrail_advanced_model"
    
    if check_model_exists(advanced_model_path):
        print("✅ Advanced model already exists")
    else:
        print("❓ Advanced transformer model not found")
        print("\nOptions:")
        print("1. Train new model (recommended, takes 10-20 min)")
        print("2. Continue with vector-only mode (faster setup)")
        print("3. Exit and download model manually")
        
        choice = input("\nChoose option (1/2/3): ").strip()
        
        if choice == "1":
            if not train_advanced_model():
                print("❌ Model training failed. You can still use vector-only mode.")
                print("   The system will work but may have reduced accuracy for edge cases.")
        elif choice == "2":
            print("⚠️  Continuing with vector-only mode")
            print("   The system will use only vector classification")
            print("   This is fast but may have reduced accuracy for edge cases")
        else:
            print("📖 Manual download instructions:")
            print("   1. Download pre-trained model from [your model hosting location]")
            print("   2. Extract to text_guardrail_advanced_model/ directory")
            print("   3. Re-run this setup script")
            sys.exit(0)
    
    # Test the system
    print("\n🧪 Testing system...")
    
    try:
        result = subprocess.run([sys.executable, "vector_text_test.py", "test message"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ System test passed!")
        else:
            print("⚠️  System test had issues but may still work")
    except Exception as e:
        print(f"⚠️  Could not run system test: {e}")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Test text: python vector_text_test.py \"your text here\"")
    print("2. Test documents: python vector_document_classifier.py --file test/test.csv")
    print("3. Run benchmarks: python speed_benchmark.py")
    print("\n📚 See README.md for full usage instructions")

if __name__ == "__main__":
    main()
