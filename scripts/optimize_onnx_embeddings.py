#!/usr/bin/env python3
"""
ONNX Optimization Script for BGE-M3 Embedding Model
Converts the BGE-M3 model to ONNX format for faster inference
"""

import os
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
import onnx
from optimum.onnxruntime import ORTModelForFeatureExtraction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_bge_m3_to_onnx():
    """
    Convert BGE-M3 model to ONNX format for optimized inference
    """
    model_name = "BAAI/bge-m3"
    output_dir = Path("models/bge-m3-onnx")
    
    logger.info(f"🚀 Starting ONNX optimization for {model_name}")
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the original model and tokenizer
        logger.info("📥 Loading original model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Convert to ONNX
        logger.info("🔄 Converting to ONNX format...")
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True,
            provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        )
        
        # Save the ONNX model
        logger.info(f"💾 Saving ONNX model to {output_dir}")
        ort_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Verify the conversion
        logger.info("✅ Verifying ONNX model...")
        onnx_model_path = output_dir / "model.onnx"
        if onnx_model_path.exists():
            onnx_model = onnx.load(str(onnx_model_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✅ ONNX model verification successful")
        else:
            logger.error("❌ ONNX model file not found")
            return False
        
        # Create a test to ensure it works
        logger.info("🧪 Testing ONNX model inference...")
        test_text = "This is a test sentence for embedding."
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = ort_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            logger.info(f"✅ ONNX inference test successful. Embedding shape: {embeddings.shape}")
        
        logger.info("🎉 ONNX optimization completed successfully!")
        logger.info(f"📁 ONNX model saved to: {output_dir.absolute()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX optimization failed: {e}")
        return False

def validate_onnx_performance():
    """
    Compare performance between original and ONNX models
    """
    import time
    
    logger.info("🏁 Starting performance comparison...")
    
    # Test text
    test_texts = [
        "This is a test sentence for performance evaluation.",
        "Another example sentence to test embedding speed.",
        "Performance testing with multiple sentences.",
        "ONNX optimization should improve inference speed.",
        "Final test sentence for comprehensive evaluation."
    ]
    
    model_name = "BAAI/bge-m3"
    onnx_path = Path("models/bge-m3-onnx")
    
    if not onnx_path.exists():
        logger.error("❌ ONNX model not found. Run optimization first.")
        return
    
    try:
        # Load ONNX model
        logger.info("📥 Loading ONNX model...")
        tokenizer = AutoTokenizer.from_pretrained(onnx_path)
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(
            onnx_path,
            provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        )
        
        # Test ONNX performance
        logger.info("⏱️ Testing ONNX inference speed...")
        start_time = time.time()
        
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = onnx_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
        
        onnx_time = time.time() - start_time
        logger.info(f"✅ ONNX inference time: {onnx_time:.4f} seconds")
        
        # Load original model for comparison
        logger.info("📥 Loading original model for comparison...")
        original_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if torch.cuda.is_available():
            original_model = original_model.cuda()
        
        # Test original performance
        logger.info("⏱️ Testing original model inference speed...")
        start_time = time.time()
        
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = original_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
        
        original_time = time.time() - start_time
        logger.info(f"✅ Original model inference time: {original_time:.4f} seconds")
        
        # Calculate speedup
        speedup = original_time / onnx_time
        logger.info(f"🚀 ONNX speedup: {speedup:.2f}x faster")
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")

if __name__ == "__main__":
    logger.info("🔧 Starting BGE-M3 ONNX Optimization")
    
    # Check if ONNX model already exists
    onnx_path = Path("models/bge-m3-onnx/model.onnx")
    
    if onnx_path.exists():
        logger.info("✅ ONNX model already exists. Running performance validation...")
        validate_onnx_performance()
    else:
        logger.info("🔄 ONNX model not found. Starting optimization...")
        if optimize_bge_m3_to_onnx():
            validate_onnx_performance()
        else:
            logger.error("❌ ONNX optimization failed") 