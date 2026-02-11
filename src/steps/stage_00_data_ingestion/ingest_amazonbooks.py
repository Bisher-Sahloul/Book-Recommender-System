import os
from pathlib import Path
import sys
import pandas as pd
import kagglehub
import zipfile
from typing import Optional, Iterable , List
from src.logger.log import logging
from src.config.configuration import AppConfiguration
from src.exception.exception_handler import AppException


class DataIngestion : 
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """
    def __init__(self , app_config = AppConfiguration()) -> None :
        """Initialize the data ingestion class."""
        """Get the amazon_book and the rating from the data folder"""
        self.data_ingestion_config = app_config.get_data_ingestion_config()

    def download_and_extract_data(self) -> None:
        """
        Downloads the latest Amazon books dataset using kagglehub.
        Returns the path to the downloaded zip file or folder.
        """
        os.makedirs(self.data_ingestion_config.Amazon_data_dir , exist_ok = True)
        
        if os.path.exists(self.data_ingestion_config.Amazon_books_data) and os.path.exists(self.data_ingestion_config.Amazon_books_rating) :
            logging.info("Dataset already exists. Skipping download.")
            return
        
        path = kagglehub.dataset_download(
                "mohamedbakhet/amazon-books-reviews",
                output_dir=self.data_ingestion_config.Amazon_data_dir,
                force_download=True
        )

        logging.info(f"Dataset downloaded to: {path}")
        
    def read_data(self) -> None : 
        Amazon_books = pd.read_csv(self.data_ingestion_config.Amazon_books_data)
        Amazon_reviews = pd.read_csv(self.data_ingestion_config.Amazon_books_rating)

        Amazon_reviews = Amazon_reviews.rename(columns={ "Id" : "ISBN" ,
                                 "review/score" : "rating" , 
                                 "review/text" : "review"
                               })
        Amazon_books = Amazon_books.merge(Amazon_reviews , how = 'left' , on = "Title")
        
        Amazon_reviews = Amazon_reviews[['ISBN' , 'User_id' , 'rating' , 'review']]
        
        Amazon_books.rename(columns={ 
                              'Title':'Book-Title' , 
                              'description' : 'Description' , 
                              'authors':'Book-Author',
                              'publisher' : 'Publisher' , 
                              'publishedDate' : 'Year-Of-Publication' ,
                              'image' : 'Image' ,
                              'categories' : 'Categories'
                             },inplace=True
                             )
        Amazon_books = Amazon_books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Description', 'Categories' , 'Image']].head(5000)
        self.df = [Amazon_books , Amazon_reviews]
        logging.info("Amazon books are ready for merging")
        
    def get_data(self) -> List: 
        return self.df
    
    def initiate_data_ingestion(self) -> List : 
        try : 
            self.download_and_extract_data()
            self.read_data()
            return self.get_data()
        except Exception as e : 
            raise AppException(e , sys) from e
