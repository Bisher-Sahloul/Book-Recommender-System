import pandas as pd 
import os 
from src.config.configuration import AppConfiguration

BOOKS = pd.read_csv(os.path.join(AppConfiguration().get_data_transformation_config().transformed_data_dir , 'current_books.csv'))

