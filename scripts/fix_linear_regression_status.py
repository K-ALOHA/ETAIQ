"""Quick script to archive LinearRegression models that are currently in Production"""
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.training.model_registry import ModelRegistryEngine
from app.core.logging import get_logger

logger = get_logger(__name__)

def main():
    registry_engine = ModelRegistryEngine()
    
    # Get all registered models
    all_models = registry_engine.list_models()
    
    for model in all_models:
        if model.model_name == "LinearRegression" and model.status == "Production":
            logger.info(f"Archiving LinearRegression v{model.version} which was incorrectly marked as Production")
            registry_engine.archive_model(model.model_name, model.version)
            
    logger.info("Done archiving LinearRegression Production models")

if __name__ == "__main__":
    main()
