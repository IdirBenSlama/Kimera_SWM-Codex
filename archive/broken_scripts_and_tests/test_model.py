try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    import torch.nn.functional as F
    import onnxruntime as ort
    import os

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)

    
    model_name = "BAAI/bge-m3"
    use_onnx = os.getenv("USE_ONNX", "1") == "1"
    onnx_model_path = os.getenv("ONNX_MODEL_PATH", "./models/bge-m3-onnx")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if use_onnx and os.path.exists(f"{onnx_model_path}/model.onnx"):
        logger.info("Loading ONNX model...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        session = ort.InferenceSession(f"{onnx_model_path}/model.onnx", providers=providers)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("ONNX model loaded successfully.")
        
        # Test inference
        test_text = "This is a test sentence."
        inputs = tokenizer(test_text, return_tensors="np", padding=True, truncation=True, max_length=512)
        outputs = session.run(None, {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        })
        logger.info(f"ONNX inference successful. Output shape: {outputs[0].shape}")
        
    else:
        logger.info("Loading Transformers model...")
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Transformers model loaded successfully.")
        
        # Test inference
        test_text = "This is a test sentence."
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
            # Normalize
            embedding = F.normalize(embedding, p=2, dim=1)
            
        logger.info(f"Transformers inference successful. Embedding shape: {embedding.shape}")
        
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Please install required packages: pip install transformers torch onnxruntime")
except Exception as e:
    logger.error(f"Error loading model: {e}")