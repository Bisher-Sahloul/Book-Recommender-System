from pathlib import Path
from typing import Optional, Iterable
import pandas as pd
import sys , os 
from src.logger.log import logging
from src.steps.stage_00_data_ingestion import merge_sources 
from src.config.configuration import AppConfiguration
from src.exception.exception_handler import AppException

class DataIngestion : 

    def __init__(self , app_config = AppConfiguration()) : 
        """
        DataIngestion Intialization
        data_ingestion_config: DataIngestionConfig 
        """
        try : 
            logging.info(f"{'*' * 20} Data Ingestion log started.{'*' * 20}")
            self.data_ingestion_config = app_config.get_data_ingestion_config()
            df = merge_sources.DataIngestion()
            df.merging_books_data()
            df.merging_reviews_data()
            [self.current_books,self.current_reviews] = df.get_data()
        except Exception as e:
            raise AppException(e , sys) from e
    def initiate_data_ingestion(self):
        try:
            self.current_books.to_csv(self.data_ingestion_config.Current_books , index=False)
            self.current_reviews.to_csv(self.data_ingestion_config.Current_reviews , index=False)
            logging.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e     
        
