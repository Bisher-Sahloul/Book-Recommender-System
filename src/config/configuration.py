import os 
import sys 
from src.logger.log import logging
from src.utils.util import read_yaml_file
from src.exception.exception_handler import AppException
from src.entity.config_entity import DataIngestionConfig , DataValidationConfig , DataTransformationConfig , ModelTrainerConfig
from src.constant import * 


class AppConfiguration:
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH):
        try : 
            self.configs_info = read_yaml_file(file_path = config_file_path) 
        except Exception as e : 
            raise AppException(e , sys) from e 
    
    def get_data_ingestion_config(self) -> DataIngestionConfig : 
        try : 
            data_ingestion_config = self.configs_info['data_ingestion_config']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
            dataset_dir = data_ingestion_config['dataset_dir']

            ingested_data_dir = os.path.join(artifacts_dir , dataset_dir , data_ingestion_config['ingested_dir'])
            raw_data_dir = os.path.join(artifacts_dir , dataset_dir , data_ingestion_config['raw_dir']) 

            Amazon_dir  = os.path.join(raw_data_dir , data_ingestion_config['Amazon_dir'])
            Openlibrary_dir = os.path.join(raw_data_dir , data_ingestion_config['Openlibrary_dir'])
            Goodreads_dir = os.path.join(raw_data_dir , data_ingestion_config['Goodreads_dir'])

            Amazon_books_data   = os.path.join(Amazon_dir , data_ingestion_config['Amazon_books_data']) 
            Amazon_Books_rating = os.path.join(Amazon_dir , data_ingestion_config['Amazon_books_rating'])

            Openlibrary_books  = os.path.join(Openlibrary_dir , data_ingestion_config['Openlibrary_books'])

            current_books = os.path.join(ingested_data_dir , data_ingestion_config['Current_books'])
            current_reviews = os.path.join(ingested_data_dir , data_ingestion_config['Current_reviews'])
            download_data_dir = os.path.join(raw_data_dir , data_ingestion_config['download_data_dir'])
            respone =  DataIngestionConfig(
                        raw_data_dir =  raw_data_dir , 
                        ingested_data_dir = ingested_data_dir , 
                        Amazon_data_dir =  Amazon_dir , 
                        download_data_dir =  download_data_dir ,
                        Openlibrary_data_dir = Openlibrary_dir , 
                        Amazon_books_data  = Amazon_books_data , 
                        Amazon_books_rating = Amazon_Books_rating , 
                        Openlibrary_books   =  Openlibrary_books , 
                        Current_books = current_books , 
                        Current_reviews = current_reviews , 
            )
            logging.info(f"Data Ingestion Config: {respone}")
            return respone
        except Exception as e:
            raise AppException(e , sys) from e  
        
    def get_data_validation_config(self) -> DataValidationConfig : 
        try : 
            data_validation_config = self.configs_info['data_validation_config']
            data_ingestion_config = self.configs_info['data_ingestion_config']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
            dataset_dir = data_ingestion_config['dataset_dir']
            ingested_data_dir = os.path.join(artifacts_dir , dataset_dir , data_ingestion_config['ingested_dir'])            
            current_books = os.path.join(ingested_data_dir , data_ingestion_config['Current_books'])
            current_reviews = os.path.join(ingested_data_dir , data_ingestion_config['Current_reviews'])


            clean_data_dir = os.path.join(artifacts_dir , dataset_dir , data_validation_config['clean_data_dir'])
            serialized_objects_dir = os.path.join(artifacts_dir , data_validation_config['serialized_objects_dir'])


            response = DataValidationConfig(
                clean_data_dir = clean_data_dir ,
                current_books_csv = current_books, 
                current_reviews_csv = current_reviews , 
                serialized_objects_dir =  serialized_objects_dir
            )
            logging.info(f"Data Validation Config: {response}")

            return response 
        except Exception as e : 
            raise AppException(e , sys) from e  
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            data_transformation_config = self.configs_info['data_transformation_config']
            data_validation_config = self.configs_info['data_validation_config']
            data_ingestion_config = self.configs_info['data_ingestion_config']
            dataset_dir = data_ingestion_config['dataset_dir']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
                      
            transformed_data_dir = os.path.join(artifacts_dir, dataset_dir, data_transformation_config['transformed_data_dir'])
            vectorstores_dir =  os.path.join(artifacts_dir , data_transformation_config['vectorstores_dir'])
            chroma_dir = os.path.join(vectorstores_dir , data_transformation_config['chroma_db_dir'])
          
            current_books = os.path.join(artifacts_dir, dataset_dir , data_validation_config['clean_data_dir'],data_transformation_config['current_books'])
            current_reviews = os.path.join(artifacts_dir, dataset_dir , data_validation_config['clean_data_dir'],data_transformation_config['current_reviews'])


            train_data = os.path.join(transformed_data_dir , data_transformation_config['train_csv'])
            test_data = os.path.join(transformed_data_dir , data_transformation_config['test_csv'])

            response = DataTransformationConfig(
                current_books_csv = current_books,
                current_reviews_csv = current_reviews,
                transformed_data_dir = transformed_data_dir,
                chroma_dir = chroma_dir , 
                train_data_csv =  train_data ,
                test_data_csv = test_data , 
            )

            logging.info(f"Data Transformation Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            model_trainer_config = self.configs_info['model_trainer_config']
            data_transformation_config = self.configs_info['data_transformation_config']
            data_ingestion_config = self.configs_info['data_ingestion_config']
            dataset_dir = data_ingestion_config['dataset_dir']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']

            transformed_data_file_dir = os.path.join(artifacts_dir, dataset_dir, data_transformation_config['transformed_data_dir'])

            trained_model_dir = os.path.join(artifacts_dir, model_trainer_config['trained_model_dir'])
            trained_model_name = model_trainer_config['trained_model_name']


            train_data = os.path.join(transformed_data_file_dir ,  data_transformation_config['train_csv'])
            test_data  = os.path.join(transformed_data_file_dir ,  data_transformation_config['test_csv'])

            response = ModelTrainerConfig(
                trained_model_dir = trained_model_dir,
                trained_model_name = trained_model_name, 
                train_csv_file = train_data , 
                test_csv_file =  test_data  
            )

            logging.info(f"Model Trainer Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e
