import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "steps", "stage_03_model_trainer", "recommenders_microsoft"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline.training_pipeline import TrainingPipeline

obj = TrainingPipeline()
obj.start_training_pipeline()