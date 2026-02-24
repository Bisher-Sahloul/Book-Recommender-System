import os 
import sys , re 
import ast 
import pickle
import pandas as pd 
from src.logger.log import logger
from src.exception.exception_handler import AppException
from src.config.configuration import AppConfiguration 

class DataValidation :
    def __init__(self , app_cofig = AppConfiguration()):
        """
        DataValidation Intialization
        data_validation_config: DataValidationConfig 
        """
        logger.info(f"{'*' * 20} Data Validation log started.{'*' * 20}")
        self.data_validation_config = app_cofig.get_data_validation_config()
    def preporcess_data(self) : 
        try : 
          current_books = pd.read_csv(self.data_validation_config.current_books_csv)
          current_reviews =  pd.read_csv(self.data_validation_config.current_reviews_csv)
          
          logger.info(f" Shape of ratings data file: {current_books.shape}")
          logger.info(f" Shape of books data file: {current_reviews.shape}")

          current_books.dropna(inplace=True)
          current_reviews = current_reviews.merge(current_books , how = 'inner' , on = 'ISBN')[['ISBN','User_id' , 'rating' , 'review']]
          current_reviews.dropna(subset=['ISBN','User_id'] , inplace=True)

          current_reviews['rating'] = current_reviews['rating'].fillna(current_reviews['rating'].mean())
          

          print(f" Shape of ratings data file: {current_books.shape}")
          print(f" Shape of books data file: {current_reviews.shape}")

          logger.info(f"Current books {current_books.head()}")
          logger.info(f"{'*' * 40}")
          logger.info(f'Current reviews {current_reviews.head()}')

          current_books = current_books.drop_duplicates(subset=["ISBN"]).reset_index(drop=True)

          current_books['word_count_per_desc'] = current_books['Description'].str.split().str.len()
          current_books = current_books[current_books['word_count_per_desc'] > 5].reset_index(drop=True)
          
          current_books['Year-Of-Publication'] = current_books['Year-Of-Publication'].apply(lambda x: re.findall(r'\d{4}', str(x))[0] if re.findall(r'\d{4}', str(x)) else None)
          current_books['Year-Of-Publication'] = current_books['Year-Of-Publication'].astype(float)
          current_books['Year-Of-Publication'].fillna(0 , inplace=True) 
          current_books = current_books[current_books['Year-Of-Publication'] > 0]

          current_books['Book-Author'] = current_books['Book-Author'].astype(str).apply(lambda x : x.strip('['))
          current_books['Book-Author'] = current_books['Book-Author'].astype(str).apply(lambda x : x.strip(']'))

          current_books['Categories'] = current_books['Categories'].astype(str).apply(lambda x : x.strip('['))
          current_books['Categories'] = current_books['Categories'].astype(str).apply(lambda x : x.strip(']'))

          current_books.drop(columns=["word_count_per_desc"] , inplace=True)

          # Saving the cleaned data for transformation
          os.makedirs(self.data_validation_config.clean_data_dir, exist_ok=True)
          current_books.to_csv(os.path.join(self.data_validation_config.clean_data_dir,'current_books_cleaned.csv'), index = False)
          current_reviews.to_csv(os.path.join(self.data_validation_config.clean_data_dir,'current_reviews_cleaned.csv'), index = False)
          logger.info(f"Saved cleaned data to {self.data_validation_config.clean_data_dir}")


        except Exception as e : 
            raise AppException(e , sys) from e 
   
    def initiate_data_validation(self):
        try:
            logger.info(f"{'='*20}Data Validation log started.{'='*20} ")
            self.preporcess_data()
            logger.info(f"{'='*20}Data Validation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e
