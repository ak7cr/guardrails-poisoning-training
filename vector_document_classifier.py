#!/usr/bin/env python3
"""
Vector-based Document Classifier - Fast document classification using embeddings
"""

import PyPDF2
import pdfplumber
import pandas as pd
from pathlib import Path
import argparse
from vector_guardrail import VectorGuardrail

class VectorDocumentClassifier:
    def __init__(self, similarity_threshold=0.75):
        """Initialize the vector-based document classifier"""
        self.guardrail = VectorGuardrail(similarity_threshold=similarity_threshold)
        print("✅ Vector Document Classifier initialized")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        text_content = []
        
        try:
            # Try pdfplumber first
            print(f"📄 Extracting text from PDF...")
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            
            if text_content:
                return " ".join(text_content)
                
        except Exception as e:
            print(f"⚠️ pdfplumber failed: {e}")
        
        try:
            # Fallback to PyPDF2
            print(f"📄 Using PyPDF2 fallback...")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            
            return " ".join(text_content)
            
        except Exception as e:
            print(f"❌ PDF extraction failed: {e}")
            return None
    
    def extract_text_from_csv(self, csv_path):
        """Extract text from CSV file"""
        try:
            print(f"📊 Reading CSV file...")
            df = pd.read_csv(csv_path)
            
            text_content = []
            
            # Add column headers
            text_content.append(f"Columns: {', '.join(df.columns.tolist())}")
            
            # Add row content
            for index, row in df.iterrows():
                row_text = " ".join([str(value) for value in row.values if pd.notna(value)])
                if row_text.strip():
                    text_content.append(row_text)
            
            combined_text = " ".join(text_content)
            print(f"📊 Extracted {len(df)} rows, {len(df.columns)} columns")
            
            return combined_text
            
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return None
    
    def classify_file(self, file_path):
        """Classify a document file using vectors"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        print(f"\n🔍 Processing file: {file_path.name}")
        print(f"📁 File size: {file_path.stat().st_size / 1024:.1f} KB")
        
        # Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == '.csv':
            text = self.extract_text_from_csv(file_path)
        else:
            return {"error": f"Unsupported file type: {file_path.suffix}"}
        
        if not text:
            return {"error": f"Could not extract text from {file_path.name}"}
        
        print(f"📝 Extracted text length: {len(text)} characters")
        print(f"📄 Text preview: {text[:200]}{'...' if len(text) > 200 else ''}")
        
        # Classify using vector approach
        result = self.guardrail.classify(text)
        result["file_name"] = file_path.name
        result["file_type"] = file_path.suffix.lower()
        result["file_size_kb"] = file_path.stat().st_size / 1024
        result["text_length"] = len(text)
        
        return result
    
    def format_file_result(self, result):
        """Format file classification result"""
        if "error" in result:
            return f"❌ {result['error']}"
        
        confidence_emoji = "🎯" if result["confidence"] > 0.9 else "⚠️" if result["confidence"] > 0.7 else "❓"
        prediction_emoji = "🚨" if result["prediction"] == "MALICIOUS" else "✅"
        method_emoji = "⚡" if result["method"] == "vector" else "🔄" if result["method"] == "hybrid" else "🤖"
        
        speed_note = "🚀 Lightning fast!" if result["method"] == "vector" else "🔄 Hybrid analysis"
        
        output = f"""
{prediction_emoji} File Classification Result
═══════════════════════════════════════════════════════════
📁 File: {result['file_name']}
📊 Prediction: {result['prediction']} {confidence_emoji}
🎯 Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)
{method_emoji} Method: {result['method'].upper()} {speed_note}

📄 File Details:
   📂 Type: {result['file_type'].upper()}
   📏 Size: {result['file_size_kb']:.1f} KB
   📝 Text Length: {result['text_length']} characters

"""
        
        if "details" in result and "matched_texts" in result["details"]:
            details = result["details"]
            output += f"🔍 Vector Analysis:\n"
            output += f"   📊 Match Type: {details['match_type'].upper()}\n"
            output += f"   🎯 Top Matches:\n"
            for i, match in enumerate(details["matched_texts"][:3], 1):
                output += f"      {i}. {match[:50]}{'...' if len(match) > 50 else ''}\n"
            
            output += f"\n   📈 Similarity Scores:\n"
            output += f"      🚨 Malicious: {details['malicious_score']:.3f}\n"
            output += f"      ✅ Benign: {details['benign_score']:.3f}\n"
            output += f"      📊 Difference: {details['score_difference']:.3f}\n"
        
        if result["method"] == "hybrid" and "fallback_details" in result:
            fb = result["fallback_details"]
            if "malicious_probability" in fb:
                output += f"\n🤖 Transformer Verification:\n"
                output += f"   🚨 Malicious: {fb['malicious_probability']:.3f}\n"
                output += f"   ✅ Benign: {fb['benign_probability']:.3f}\n"
        
        return output

def main():
    parser = argparse.ArgumentParser(description="Vector-based Document Classifier")
    parser.add_argument("--file", "-f", type=str, help="Path to document file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold")
    parser.add_argument("--batch", "-b", nargs="+", help="Batch process multiple files")
    
    args = parser.parse_args()
    
    print("🚀 Vector-based Document Classifier")
    print("🔥 Lightning Fast • 🎯 High Accuracy • 🤖 AI-Powered")
    print("=" * 60)
    
    # Initialize classifier
    classifier = VectorDocumentClassifier(similarity_threshold=args.threshold)
    
    if args.file:
        # Single file mode
        result = classifier.classify_file(args.file)
        print(classifier.format_file_result(result))
        
    elif args.batch:
        # Batch mode
        print(f"\n📦 Batch processing {len(args.batch)} files...")
        
        for i, file_path in enumerate(args.batch, 1):
            print(f"\n[{i}/{len(args.batch)}] Processing: {Path(file_path).name}")
            result = classifier.classify_file(file_path)
            print(classifier.format_file_result(result))
            
            if i < len(args.batch):
                print("─" * 60)
        
    elif args.interactive:
        # Interactive mode
        print("\n🔄 Interactive Mode - Enter file paths to classify")
        print("💡 Supported: PDF, CSV")
        print("💡 Type 'quit' to exit\n")
        
        while True:
            try:
                file_path = input("📁 Enter file path: ").strip()
                
                if file_path.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not file_path:
                    continue
                
                result = classifier.classify_file(file_path)
                print(classifier.format_file_result(result))
                print("\n" + "─" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    
    else:
        # Auto-detect files in current directory
        print("\n🎯 Auto-detection mode - Processing available files")
        
        current_dir = Path(".")
        test_files = []
        
        for ext in [".pdf", ".csv"]:
            test_files.extend(list(current_dir.glob(f"*{ext}")))
        
        if test_files:
            print(f"📁 Found {len(test_files)} files to process:")
            
            for i, file_path in enumerate(test_files[:5], 1):  # Limit to 5 files
                print(f"\n[{i}] {file_path.name}")
                result = classifier.classify_file(file_path)
                print(classifier.format_file_result(result))
                
                if i < min(len(test_files), 5):
                    print("─" * 60)
                    
            if len(test_files) > 5:
                print(f"\n💡 Showing first 5 files. Total found: {len(test_files)}")
                print("💡 Use --batch *.pdf *.csv to process all files")
        else:
            print("📁 No PDF or CSV files found in current directory")
            print(f"\n💡 Usage examples:")
            print(f"   python vector_document_classifier.py --file document.pdf")
            print(f"   python vector_document_classifier.py --interactive")
            print(f"   python vector_document_classifier.py --batch *.pdf *.csv")

if __name__ == "__main__":
    main()
