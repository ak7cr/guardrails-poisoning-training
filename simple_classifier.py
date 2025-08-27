#!/usr/bin/env python3
"""
Simple Document Classifier
Clean implementation for PDF/CSV malicious content detection
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import PyPDF2
import pdfplumber
import pandas as pd
from production_guardrail import ProductionSmartGuardrail

class SimpleDocumentClassifier:
    def __init__(self):
        print("🔍 Initializing Simple Document Classifier...")
        
        # Use the existing trained production guardrail
        self.classifier = ProductionSmartGuardrail()
        print("✅ Classifier ready!")
    
    def classify_document(self, file_path):
        """Classify a document as malicious or normal"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return None
            
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            
            print(f"\n📄 Analyzing: {filename}")
            print("-" * 50)
            
            # Extract text content
            if file_ext == '.pdf':
                content = self._extract_pdf_text(file_path)
            elif file_ext == '.csv':
                content = self._extract_csv_text(file_path)
            elif file_ext == '.txt':
                content = self._extract_txt_text(file_path)
            else:
                print(f"❌ Unsupported file type: {file_ext}")
                return None
            
            if not content or not content.strip():
                print("⚠️  Warning: No readable content found")
                return {"classification": "UNKNOWN", "confidence": 0.0}
            
            # Classify using production guardrail
            result = self.classifier.filter_content(content)
            
            # Convert to simple classification
            safety_score = result.safety_score
            is_modified = result.is_modified
            
            # Classification logic
            if safety_score >= 0.85:
                classification = "BENIGN"
                confidence = safety_score
                status_icon = "✅"
            elif safety_score >= 0.70:
                classification = "BENIGN"
                confidence = safety_score
                status_icon = "🟢"
            elif safety_score >= 0.50:
                classification = "SUSPICIOUS"
                confidence = 1 - safety_score
                status_icon = "🟡"
            else:
                classification = "MALICIOUS"
                confidence = 1 - safety_score
                status_icon = "🔴"
            
            print(f"📊 Safety Score: {safety_score:.3f}")
            print(f"🔍 Content Modified: {'Yes' if is_modified else 'No'}")
            print(f"{status_icon} Classification: {classification}")
            print(f"📈 Confidence: {confidence:.3f}")
            
            # File info
            file_size = os.path.getsize(file_path)
            print(f"\n📋 File Info:")
            print(f"   Size: {file_size / 1024:.1f} KB")
            print(f"   Type: {file_ext.upper()}")
            print(f"   Content Length: {len(content)} characters")
            
            if is_modified:
                print(f"\n⚠️  Potential Issues Detected:")
                print(f"   Content was filtered for safety concerns")
            
            return {
                "classification": classification,
                "confidence": confidence,
                "safety_score": safety_score,
                "is_modified": is_modified,
                "file_size": file_size,
                "content_length": len(content)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing document: {e}")
            return None
    
    def _extract_pdf_text(self, file_path):
        """Extract text from PDF"""
        try:
            with pdfplumber.open(file_path) as pdf:
                text_content = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
                return "\n".join(text_content)
        except:
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = []
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
                    return "\n".join(text_content)
            except Exception as e:
                print(f"⚠️  PDF extraction error: {e}")
                return ""
    
    def _extract_csv_text(self, file_path):
        """Extract text from CSV"""
        try:
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Unable to decode CSV file")
            
            text_parts = []
            text_parts.append(f"CSV Data: {len(df)} rows, {len(df.columns)} columns")
            text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
            sample_text = df.head(10).to_string()
            text_parts.append(sample_text)
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"⚠️  CSV extraction error: {e}")
            return ""
    
    def _extract_txt_text(self, file_path):
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin1') as file:
                    return file.read()
            except Exception as e:
                print(f"⚠️  TXT extraction error: {e}")
                return ""
    
    def batch_classify(self, file_paths):
        """Classify multiple documents"""
        print(f"\n🔍 Batch Classification ({len(file_paths)} files)")
        print("=" * 60)
        
        results = []
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n[{i}/{len(file_paths)}] Processing: {os.path.basename(file_path)}")
            result = self.classify_document(file_path)
            if result:
                results.append({
                    "file": os.path.basename(file_path),
                    "path": file_path,
                    **result
                })
        
        # Summary
        if results:
            print(f"\n📊 Batch Summary:")
            malicious_count = sum(1 for r in results if r["classification"] == "MALICIOUS")
            suspicious_count = sum(1 for r in results if r["classification"] == "SUSPICIOUS")
            benign_count = len(results) - malicious_count - suspicious_count
            
            print(f"   🔴 Malicious: {malicious_count}")
            print(f"   🟡 Suspicious: {suspicious_count}")
            print(f"   ✅ Benign: {benign_count}")
        
        return results
    
    def run_interactive(self):
        """Interactive classification mode"""
        print("\n" + "="*60)
        print("🔍 SIMPLE DOCUMENT CLASSIFIER")
        print("🛡️  Powered by Production Guardrail")
        print("="*60)
        print("Commands:")
        print("  classify <file_path>     - Classify single document")
        print("  batch <folder_path>      - Classify all docs in folder")
        print("  help                     - Show help")
        print("  quit                     - Exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🔍 > ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                elif user_input.lower() == 'help':
                    self.show_help()
                    
                elif user_input.startswith('classify '):
                    file_path = user_input[9:].strip().strip('\'"')
                    self.classify_document(file_path)
                    
                elif user_input.startswith('batch '):
                    folder_path = user_input[6:].strip().strip('\'"')
                    self.classify_folder(folder_path)
                    
                else:
                    # Try to treat input as file path
                    if os.path.exists(user_input.strip('\'"')):
                        self.classify_document(user_input.strip('\'"'))
                    else:
                        print("❌ Unknown command. Type 'help' for available commands.")
                        
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def classify_folder(self, folder_path):
        """Classify all documents in a folder"""
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return
        
        supported_extensions = ['.pdf', '.csv', '.txt']
        file_paths = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                file_paths.append(os.path.join(folder_path, file))
        
        if not file_paths:
            print(f"⚠️  No supported documents found in: {folder_path}")
            return
        
        self.batch_classify(file_paths)
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*60)
        print("🔍 SIMPLE DOCUMENT CLASSIFIER HELP")
        print("🛡️  Powered by Production Guardrail")
        print("="*60)
        print("COMMANDS:")
        print("  classify <file_path>     - Analyze single document")
        print("  batch <folder_path>      - Analyze all documents in folder")
        print("  help                     - Show this help")
        print("  quit                     - Exit program")
        print("\nEXAMPLES:")
        print('  classify "document.pdf"')
        print('  batch "C:\\suspicious_files\\"')
        print("\nCLASSIFICATION LEVELS:")
        print("  🔴 MALICIOUS - High risk content detected")
        print("  🟡 SUSPICIOUS - Potentially harmful content")
        print("  🟢 BENIGN - Safe content (medium confidence)")
        print("  ✅ BENIGN - Safe content (high confidence)")
        print("\nFEATURES:")
        print("  🛡️  Production-grade security filtering")
        print("  📊 Confidence scores and safety metrics")
        print("  📄 PDF, CSV, TXT support")
        print("  ⚡ Fast classification")
        print("=" * 60)

def main():
    """Main entry point"""
    try:
        classifier = SimpleDocumentClassifier()
        
        # Check if file path provided as argument
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            classifier.classify_document(file_path)
        else:
            # Interactive mode
            classifier.run_interactive()
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
