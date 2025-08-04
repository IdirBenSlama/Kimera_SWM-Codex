import logging
import traceback

# Configure logging to a dedicated file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    filename='startup_error.log',
    filemode='w'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

def diagnose_startup():
    """Attempt to import the FastAPI app and log any errors."""
    logger.info("Starting diagnostic import of backend.api.main:app...")
    try:
        from src.api.main import app
        logger.info("✅ SUCCESS: The FastAPI 'app' was imported successfully.")
        logger.info(f"   App type: {type(app)}")
        logger.info("   This suggests the issue may lie with the uvicorn server runner.")
        
    except Exception as e:
        logger.error("❌ FAILED: A critical error occurred during application import.")
        logger.error(f"   Error Type: {type(e).__name__}")
        logger.error(f"   Error Message: {e}")
        
        # Log the full traceback to the file
        tb_str = traceback.format_exc()
        logging.getLogger().handlers[0].stream.write(f"--- TRACEBACK ---\n{tb_str}\n")
        logger.error("   Full traceback has been written to startup_error.log")

if __name__ == "__main__":
    diagnose_startup() 