import sys
import time
import threading
import uvicorn
from src.config import API_HOST, API_PORT, UI_PORT
from src.utils.logger import logger
from src.ui.app import demo

def run_api_server():
    """
    Spins up the FastAPI backend server using Uvicorn.
    """
    logger.info(f"Starting API Gateway at http://{API_HOST}:{API_PORT}")
    try:
        uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, log_level="warning")
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        sys.exit(1)

def main():
    logger.info("Initializing RecoStream unified environment...")
    
    # 1. Start the API Backend in a background daemon thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Give the API server 3 seconds to spin up, fit preprocessing, and pre-train models
    logger.info("Waiting for API server to pre-train models and start up...")
    time.sleep(3.5)
    
    # 2. Start the Gradio Web UI in the main thread
    logger.info(f"Launching Gradio interface at http://{API_HOST}:{UI_PORT}")
    try:
        demo.launch(server_name=API_HOST, server_port=UI_PORT, share=False)
    except KeyboardInterrupt:
        logger.info("Shutting down RecoStream services gracefully.")
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
