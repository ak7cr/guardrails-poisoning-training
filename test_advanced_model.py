#!/usr/bin/env python3
"""
Advanced Model Tester - Test PDF and CSV files with the trained Focal Loss model
"""

import torch
import pandas as pd
import PyPDF2
import pdfplumber
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import argparse
import json

class AdvancedDocumentClassifier:
    def __init__(self, model_path="./text_guardrail_advanced_model"):
        """Initialize the classifier with the trained model"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            print(f"🔧 Loading model from {self.model_path}...")
            print(f"🎮 Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"📋 Model config: {self.model.config.num_labels} classes")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"💡 Make sure the model exists at: {self.model_path}")
            sys.exit(1)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file using multiple methods"""
        text_content = []
        
        try:
            # Method 1: Try pdfplumber first (better for complex PDFs)
            print(f"📄 Extracting text from PDF using pdfplumber...")
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"Page {page_num}: {page_text}")
            
            if text_content:
                return " ".join(text_content)
                
        except Exception as e:
            print(f"⚠️ pdfplumber failed: {e}")
        
        try:
            # Method 2: Fallback to PyPDF2
            print(f"📄 Extracting text from PDF using PyPDF2...")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"Page {page_num}: {page_text}")
            
            return " ".join(text_content)
            
        except Exception as e:
            print(f"❌ Both PDF extraction methods failed: {e}")
            return None
    
    def extract_text_from_csv(self, csv_path):
        """Extract text from CSV file"""
        try:
            print(f"📊 Reading CSV file...")
            df = pd.read_csv(csv_path)
            
            # Combine all text content from the CSV
            text_content = []
            
            # Add column headers
            text_content.append(f"Columns: {', '.join(df.columns.tolist())}")
            
            # Add all cell content as text
            for index, row in df.iterrows():
                row_text = " ".join([str(value) for value in row.values if pd.notna(value)])
                if row_text.strip():
                    text_content.append(f"Row {index + 1}: {row_text}")
            
            combined_text = " ".join(text_content)
            print(f"📊 Extracted {len(df)} rows, {len(df.columns)} columns")
            
            return combined_text
            
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return None
    
    def classify_text(self, text, max_length=512):
        """Classify text using the trained model"""
        if not text or not text.strip():
            return None, None, "No text to classify"
        
        try:
            # Tokenize the text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Extract probabilities
            malicious_prob = probabilities[0][1].item()
            benign_prob = probabilities[0][0].item()
            
            # Determine classification
            predicted_class = "MALICIOUS" if malicious_prob > 0.5 else "BENIGN"
            confidence = max(malicious_prob, benign_prob)
            
            return predicted_class, confidence, {
                "malicious_probability": malicious_prob,
                "benign_probability": benign_prob,
                "text_length": len(text),
                "truncated": len(text) > max_length
            }
            
        except Exception as e:
            return None, None, f"Classification error: {e}"
    
    def classify_file(self, file_path):
        """Classify a PDF or CSV file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return f"❌ File not found: {file_path}"
        
        print(f"\n🔍 Processing file: {file_path.name}")
        print(f"📁 File size: {file_path.stat().st_size / 1024:.1f} KB")
        
        # Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == '.csv':
            text = self.extract_text_from_csv(file_path)
        else:
            return f"❌ Unsupported file type: {file_path.suffix}"
        
        if not text:
            return f"❌ Could not extract text from {file_path.name}"
        
        print(f"📝 Extracted text length: {len(text)} characters")
        print(f"📄 Text preview: {text[:200]}{'...' if len(text) > 200 else ''}")
        
        # Classify the text
        prediction, confidence, details = self.classify_text(text)
        
        if prediction is None:
            return f"❌ Classification failed: {details}"
        
        # Format results
        result = {
            "file": file_path.name,
            "file_type": file_path.suffix.lower(),
            "prediction": prediction,
            "confidence": confidence,
            "details": details
        }
        
        return result
    
    def format_result(self, result):
        """Format classification result for display"""
        if isinstance(result, str):
            return result
        
        confidence_emoji = "🎯" if result["confidence"] > 0.9 else "⚠️" if result["confidence"] > 0.7 else "❓"
        prediction_emoji = "🚨" if result["prediction"] == "MALICIOUS" else "✅"
        
        output = f"""
{prediction_emoji} Classification Result for: {result['file']}
═══════════════════════════════════════════════════════════
📊 Prediction: {result['prediction']} {confidence_emoji}
🎯 Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)
📄 File Type: {result['file_type'].upper()}

📈 Detailed Probabilities:
   🚨 Malicious: {result['details']['malicious_probability']:.3f} ({result['details']['malicious_probability']*100:.1f}%)
   ✅ Benign:    {result['details']['benign_probability']:.3f} ({result['details']['benign_probability']*100:.1f}%)

📝 Text Analysis:
   📏 Length: {result['details']['text_length']} characters
   ✂️ Truncated: {'Yes' if result['details']['truncated'] else 'No'}
"""
        return output

def main():
    parser = argparse.ArgumentParser(description="Test PDF/CSV files with advanced guardrail model")
    parser.add_argument("--file", "-f", type=str, help="Path to PDF or CSV file to classify")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--model", "-m", type=str, default="./text_guardrail_advanced_model", 
                       help="Path to trained model")
    
    args = parser.parse_args()
    
    print("🚀 Advanced Document Guardrail Classifier")
    print("=" * 60)
    
    # Initialize classifier
    classifier = AdvancedDocumentClassifier(args.model)
    
    if args.file:
        # Single file mode
        result = classifier.classify_file(args.file)
        print(classifier.format_result(result))
        
    elif args.interactive:
        # Interactive mode
        print("\n🔄 Interactive Mode - Enter file paths to classify")
        print("💡 Supported formats: PDF, CSV")
        print("💡 Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                file_path = input("📁 Enter file path (or 'quit'): ").strip()
                
                if file_path.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not file_path:
                    continue
                
                result = classifier.classify_file(file_path)
                print(classifier.format_result(result))
                print("\n" + "─" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    else:
        # Demo mode with existing files
        print("\n🎯 Demo Mode - Testing with available files")
        print("💡 Use --interactive for interactive mode or --file <path> for single file")
        
        # Look for test files in current directory
        test_files = []
        current_dir = Path(".")
        
        for ext in [".pdf", ".csv"]:
            test_files.extend(list(current_dir.glob(f"*{ext}")))
        
        if test_files:
            print(f"\n📁 Found {len(test_files)} test files:")
            for i, file_path in enumerate(test_files[:5], 1):  # Limit to 5 files
                print(f"   {i}. {file_path.name}")
                result = classifier.classify_file(file_path)
                print(classifier.format_result(result))
                print("─" * 60)
        else:
            print(f"\n📁 No PDF or CSV files found in current directory")
            print(f"💡 Available options:")
            print(f"   python test_advanced_model.py --file <path>")
            print(f"   python test_advanced_model.py --interactive")

if __name__ == "__main__":
    main()
