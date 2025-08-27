#!/usr/bin/env python3
"""
Quick Vector Text Classifier
Lightning-fast text classification using vector embeddings
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_guardrail import VectorGuardrail

def main():
    print("🚀 Quick Vector Text Classifier")
    print("🔥 Lightning Fast • 🎯 High Accuracy • 🤖 AI-Powered")
    print("=" * 60)
    
    # Initialize vector guardrail
    guardrail = VectorGuardrail()
    
    if len(sys.argv) > 1:
        # Text provided as argument
        text = " ".join(sys.argv[1:])
        result = guardrail.classify(text)
        
        print(f"\n📝 Text: {text}")
        print("=" * 60)
        print(f"📊 Prediction: {result['prediction']} {'🚨' if result['prediction'] == 'MALICIOUS' else '✅'}")
        print(f"🎯 Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
        print(f"🔄 Method: {result['method']}")
        if result.get('details'):
            print(f"📈 Vector Scores: Mal={result['details'].get('malicious_score', 0):.3f}, Ben={result['details'].get('benign_score', 0):.3f}")
    else:
        # Interactive mode
        print("\n🔄 Interactive Mode - Enter text to classify")
        print("💡 Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                text = input("📝 Enter text: ").strip()
                if text.lower() in ['quit', 'exit', '']:
                    break
                
                result = guardrail.classify(text)
                
                print("─" * 60)
                print(f"📊 Prediction: {result['prediction']} {'🚨' if result['prediction'] == 'MALICIOUS' else '✅'}")
                print(f"🎯 Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
                print(f"🔄 Method: {result['method']}")
                if result.get('details'):
                    print(f"📈 Vector Scores: Mal={result['details'].get('malicious_score', 0):.3f}, Ben={result['details'].get('benign_score', 0):.3f}")
                print("─" * 60)
                print()
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    
    print("\n👋 Thanks for using Vector Text Classifier!")

if __name__ == "__main__":
    main()
