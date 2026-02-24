import logging
import os
from pathlib import Path
from typing import Optional, Iterable
from src.config.configuration import AppConfiguration
from src.logger.log import logger
import pandas as pd
from typing import List

from src.steps.stage_00_data_ingestion import ingest_amazonbooks
from src.steps.stage_00_data_ingestion import ingest_openlibrary


class DataIngestion : 
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """
    def __init__(self) -> None :
        """Initialize the data ingestion class."""
        [self.Amazon_books , self.Amazon_reviews] = ingest_amazonbooks.DataIngestion().get_data()
        self.openlibrary_books = ingest_openlibrary.DataIngestion(pages=range(67,68)).get_data()
        self.current_books = pd.DataFrame()
        self.current_reviews = pd.DataFrame()
    def merging_books_data(self) -> None  : 
        self.current_books = pd.concat([self.Amazon_books ,self.openlibrary_books] , axis=0).reset_index(drop=True)
    def merging_reviews_data(self) -> None : 
        print(self.Amazon_reviews.shape)
        self.Amazon_reviews = self.Amazon_reviews[['ISBN','User_id' , 'rating' , 'review']]
        self.Amazon_reviews = pd.concat([self.Amazon_reviews , self.support_ratings] , axis = 0).reset_index(drop=True)
        self.Amazon_reviews.drop_duplicates(subset=['ISBN','User_id'] , inplace=True)
        self.current_reviews = self.current_books.merge(self.Amazon_reviews , how='inner' , on = 'ISBN')[['ISBN','User_id' , 'rating' , 'review']]
        print(self.current_reviews.shape)

    def get_data(self)  : 
        return [self.current_books,self.current_reviews]


