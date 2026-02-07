from src.steps.stage_00_data_ingestion.Ingest_step import DataIngestion
from src.steps.stage_01_data_validation.validate_step import DataValidation
from src.steps.stage_02_data_transformation.transform_step import DataTransformation
from src.steps.stage_03_model_trainer.train_step import ModelTrainer


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_training_pipeline(self):
        """
        Starts the training pipeline
        :return: none
        """
        self.data_ingestion.initiate_data_ingestion()
        self.data_validation.initiate_data_validation()
        self.data_transformation.initiate_data_transformation()
        self.model_trainer.initiate_model_trainer() 