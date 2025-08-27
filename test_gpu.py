#!/usr/bin/env python3
"""
GPU Test Script - Check if CUDA is working properly
"""

import torch
import time

def test_gpu():
    print("🔍 GPU Test Script")
    print("=" * 50)
    
    # Basic CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test tensor operations
        print("\n🧪 Testing tensor operations...")
        
        # CPU test
        print("Testing CPU...")
        start_time = time.time()
        x_cpu = torch.randn(1000, 1000)
        y_cpu = torch.randn(1000, 1000)
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.4f} seconds")
        
        # GPU test
        print("Testing GPU...")
        device = torch.device("cuda:0")
        start_time = time.time()
        x_gpu = torch.randn(1000, 1000, device=device)
        y_gpu = torch.randn(1000, 1000, device=device)
        z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.4f} seconds")
        
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x")
        
        # Memory usage
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.3f} GB")
        
        return True
    else:
        print("❌ CUDA not available!")
        return False

def test_model_loading():
    if not torch.cuda.is_available():
        print("❌ CUDA not available for model test")
        return
        
    print("\n🤖 Testing model loading on GPU...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "distilbert-base-uncased"  # Use smaller model for test
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Move to GPU
        device = torch.device("cuda:0")
        model = model.to(device)
        
        print(f"✅ Model loaded on GPU")
        print(f"Model parameters: {model.num_parameters():,}")
        print(f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        
        # Test inference
        text = "This is a test sentence"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        print(f"✅ GPU inference successful")
        print(f"Output shape: {outputs.logits.shape}")
        print(f"Output device: {outputs.logits.device}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")

if __name__ == "__main__":
    if test_gpu():
        test_model_loading()
    
    print("\n🏁 GPU test complete!")
