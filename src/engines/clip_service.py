import os
import sys
import logging

logger = logging.getLogger(__name__)

# Check PyTorch version compatibility for CVE-2025-32434
def is_torch_safe():
    """Check if PyTorch version is safe from CVE-2025-32434"""
    try:
        import torch
        version_parts = torch.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        
        # PyTorch 2.6+ is required for safe torch.load
        if major > 2 or (major == 2 and minor >= 6):
            return True
        else:
            logger.warning(f"PyTorch {torch.__version__} detected - CVE-2025-32434 vulnerability present")
            return False
    except Exception as e:
        logger.warning(f"Could not check PyTorch version: {e}")
        return False

# Use lightweight fallback if torch is unsafe or LIGHTWEIGHT_CLIP is set
use_lightweight = (
    os.getenv("LIGHTWEIGHT_CLIP", "0") == "1" or 
    not is_torch_safe()
)

if use_lightweight:
    logger.info("Using lightweight CLIP fallback (no heavy model loading)")
    CLIPProcessor = None  # type: ignore
    CLIPModel = None  # type: ignore
    torch = None  # type: ignore
else:
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        logger.info("Heavy CLIP models available - using full functionality")
    except Exception as e:  # pragma: no cover - allow tests without heavy deps
        logger.warning(f"Could not import CLIP models: {e}")
        CLIPProcessor = None  # type: ignore
        CLIPModel = None  # type: ignore
        torch = None  # type: ignore

from PIL import Image
import numpy as np
from ..utils.config import get_api_settings
from ..config.settings import get_settings


class CLIPService:
    """Wrapper around OpenAI's CLIP model with graceful fallbacks and security considerations."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        # Only load models if safe and available
        if CLIPModel is not None and torch is not None:
            try:
                logger.info(f"Loading CLIP model: {model_name} on {self.device}")
                self.model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                logger.info("✅ CLIP model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load CLIP model: {e}")
                logger.info("🔄 Falling back to lightweight mode")
                self.model = None
                self.processor = None
        else:  # pragma: no cover - lightweight fallback
            logger.info("🪶 CLIP service running in lightweight mode (security/performance)")

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get image embedding with fallback to zeros if model unavailable"""
        if self.model is None or self.processor is None or torch is None:
            logger.debug("Using fallback image embedding (zeros)")
            return np.zeros(512)
            
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            return np.zeros(512)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding with fallback to zeros if model unavailable"""
        if self.model is None or self.processor is None or torch is None:
            logger.debug("Using fallback text embedding (zeros)")
            return np.zeros(512)
            
        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return np.zeros(512)

    def is_available(self) -> bool:
        """Check if CLIP service is fully functional"""
        return self.model is not None and self.processor is not None

    def get_status(self) -> dict:
        """Get service status information"""
        return {
            "available": self.is_available(),
            "device": self.device,
            "model_name": self.model_name if self.is_available() else "lightweight_fallback",
            "torch_safe": is_torch_safe(),
            "lightweight_mode": not self.is_available()
        }


# Create service instance with safety checks
try:
    clip_service = CLIPService()
    logger.info(f"CLIP service initialized: {clip_service.get_status()}")
except Exception as e:
    logger.error(f"Failed to initialize CLIP service: {e}")
    # Create minimal fallback service
    clip_service = CLIPService()

