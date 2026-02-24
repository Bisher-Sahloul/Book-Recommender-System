import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "steps", "stage_03_model_trainer", "recommenders_microsoft"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline.training_pipeline import TrainingPipeline
from src.logger.log import setup_logging
setup_logging()

obj = TrainingPipeline()
obj.start_training_pipeline()