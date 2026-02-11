import os
import sys
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from src.logger.log import logging
from src.config.configuration import AppConfiguration
from src.exception.exception_handler import AppException

from langchain_classic.schema import Document

from sklearn.model_selection import train_test_split
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class DataTransformation:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.data_transformation_config = app_config.get_data_transformation_config()
            self.data_validation_config= app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e
    
    def most_popular_book(self , current_books: pd.DataFrame) -> None :
        try :
            logging.info(f"{'='*20}Making most popular books list.{'='*20}")
            most_popular_books = current_books
            most_popular_books = most_popular_books.sort_values(by="rating" , ascending=False).reset_index(drop=True)
            logging.info(f"Most popular books : {most_popular_books}")
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            pickle.dump(most_popular_books , open(os.path.join(self.data_transformation_config.transformed_data_dir,"most_popular_books.pkl"),'wb'))
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            pickle.dump(most_popular_books , open(os.path.join(self.data_validation_config.serialized_objects_dir, "book_pivot.pkl"),'wb'))
            logging.info(f"Save most popular book : {self.data_validation_config.serialized_objects_dir}")
        except Exception as e:
            raise AppException(e, sys) from e
        
    def make_piovt_table(self , current_reviews) -> None : 
        try:
            logging.info(f"{'='*20}Making pivot table{'='*20}")
            pt = current_reviews[['ISBN' , 'User_id' , 'rating']].pivot_table(index='User_id', columns='ISBN' , values='rating' , fill_value=0)
            logging.info(f" Shape of book pivot table: {pt.shape}")            
            #saving pivot table data
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            pickle.dump(pt,open(os.path.join(self.data_transformation_config.transformed_data_dir,"piovt_table_data.pkl"),'wb'))
            logging.info(f"Saved pivot table sparse data to {self.data_transformation_config.transformed_data_dir}")
            #keeping books name
            book_ISBNs = pt.index
            #saving book_names objects for web app
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            pickle.dump(book_ISBNs,open(os.path.join(self.data_validation_config.serialized_objects_dir, "book_ISBNs.pkl"),'wb'))
            logging.info(f"Saved book_ISBNs serialization object to {self.data_validation_config.serialized_objects_dir}")
            #saving book_pivot objects for web app
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            pickle.dump(pt,open(os.path.join(self.data_validation_config.serialized_objects_dir, "piovt_table_data.pkl"),'wb'))
            logging.info(f"Saved book_pivot serialization object to {self.data_validation_config.serialized_objects_dir}")
        except Exception as e: 
            raise AppException(e , sys) from e 

    def make_train_test_dataset_for_model(self , current_reviews) -> None : 
        try : 
            logging.info("Start splitting data.\n")
            os.makedirs(self.data_transformation_config.transformed_data_dir , exist_ok = True)
            train, test = train_test_split(current_reviews.values, test_size=0.2, random_state = 16)
            train = pd.DataFrame(train, columns = current_reviews.columns)
            test = pd.DataFrame(test, columns = current_reviews.columns)
            train = train.rename(columns={'User_id': 'userID', 'ISBN': 'itemID'})
            test = test.rename(columns={'User_id': 'userID', 'ISBN': 'itemID'})
            train.to_csv(self.data_transformation_config.train_data_csv , index=False)
            test.to_csv(self.data_transformation_config.test_data_csv  , index = False)
            
        except Exception as e  : 
            raise AppException(e , sys) from e 

        
    def make_vector_database(self , current_books:pd.DataFrame , current_reviews:pd.DataFrame) -> None :
        try:
            os.makedirs(self.data_transformation_config.vectorstores_dir , exist_ok = True)
            documents = []
            for _, row in current_books.iterrows():
                documents.append(
                Document(
                    page_content=row["Description"],   # أو description
                    metadata={
                            "ISBN": row["ISBN"],
                            "Book_Title": row["Book-Title"],
                            "Book_Author": row["Book-Author"],
                            "Year_Of_Publication": row["Year-Of-Publication"],
                            "Publisher": row["Publisher"] ,
                            "Description" : row["Description"] , 
                            "Categories" : row["Categories"] , 
                            "Image" : row["Image"] , 
                            "rating" : row["rating"]
                        }
                    )
                )
            current_books.to_csv(os.path.join(self.data_transformation_config.transformed_data_dir , 'current_books.csv') , index=False)
            # current_books["ISBN_desc"] = current_books["ISBN"] + " " + current_books["Description"]
            # current_books["ISBN_desc"].to_csv(os.path.join(self.data_validation_config.serialized_objects_dir,'ISBN_desc.txt') , sep = "\n", index = False , header=False)
            # loader = TextLoader(os.path.join(self.data_validation_config.serialized_objects_dir,'ISBN_desc.txt') , encoding="utf-8")
            # raw_documents = loader.load()
            # text_splitter = CharacterTextSplitter(separator="\n", chunk_overlap=0)
            # documents = text_splitter.split_documents(raw_documents)
            persist_dir = self.data_transformation_config.chroma_dir
            # create / persist when building:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents, embeddings, persist_directory = persist_dir)
        except Exception as e : 
            raise AppException(e , sys) from e 


    def get_data_transformer(self):
        try:
            logging.info(f"{"="*20}Transforming data.{"="*20}\n\n")
            current_reviews = pd.read_csv(self.data_transformation_config.current_reviews_csv)
            current_books = pd.read_csv(self.data_transformation_config.current_books_csv)
            
            
            temp_current_reviews = current_reviews 

            named_aggs = {
                'rating'  : ('rating', 'mean'),
                'num_ratings' : ('rating', 'count')
            }
            temp_current_reviews = temp_current_reviews.groupby('ISBN').agg(**named_aggs).reset_index()
            temp_current_reviews_1 = temp_current_reviews [temp_current_reviews['num_ratings'] >= 50]
            temp_current_reviews_1.reset_index(drop=True , inplace=True)
            current_books_1 = current_books.merge(temp_current_reviews_1 , how = 'inner' , on = "ISBN")
            current_books_1.drop(columns=['num_ratings'] , inplace=True)
            current_books_1['rating'] = current_books_1['rating'].round(1)


            temp_current_reviews.drop(columns=['num_ratings'] , inplace=True)
            temp_current_reviews['rating'] = temp_current_reviews['rating'].round(1)
            current_books_2 = current_books.merge(temp_current_reviews , how = 'inner' , on = "ISBN")


            self.most_popular_book(current_books = current_books_1)
            self.make_piovt_table(current_reviews = current_reviews)
            self.make_vector_database(current_books = current_books_2 , current_reviews = current_reviews[['ISBN' , 'rating']])
            self.make_train_test_dataset_for_model(current_reviews = current_reviews[['ISBN' , 'User_id' , 'rating']])
           
        except Exception as e:
            raise AppException(e, sys) from e

    

    def initiate_data_transformation(self):
        try:
            logging.info(f"{'='*20}Data Transformation log started.{'='*20} ")
            self.get_data_transformer()
            logging.info(f"{'='*20}Data Transformation log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e


