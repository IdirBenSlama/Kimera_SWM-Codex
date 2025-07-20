import torch

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()
    logger.info(f"Current GPU: {torch.cuda.current_device()
    logger.info(f"GPU Name: {torch.cuda.get_device_name(0)
else:
    logger.info("PyTorch cannot find a compatible CUDA device. Please check your NVIDIA driver and CUDA Toolkit installation.")