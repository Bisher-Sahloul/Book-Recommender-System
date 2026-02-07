from collections import namedtuple

DataIngestionConfig = namedtuple(
    "DatasetConfig" , [
                       "raw_data_dir" ,
                       "ingested_data_dir" ,
                       "Amazon_data_dir" , 
                       "Openlibrary_data_dir" , 
                       "Amazon_books_data" , 
                       "Amazon_books_rating" , 
                       "Openlibrary_books" , 
                       "Current_books", 
                       "Current_reviews"
                      ]
)


DataValidationConfig = namedtuple(
    "DataValidationConfig" , [
        "clean_data_dir" , 
        "current_books_csv" , 
        "current_reviews_csv" ,
        "serialized_objects_dir"
    ] 
)


DataTransformationConfig = namedtuple(
    "DataTransformationConfig", [
        "current_books_csv",
        "current_reviews_csv" ,
        "transformed_data_dir",
        "chroma_dir" , 
        "train_data_csv" , 
        "test_data_csv" 
    ]
)

ModelTrainerConfig = namedtuple(
    "ModelTrainerConfig", [
        "trained_model_dir",
        "trained_model_name",
        "train_csv_file" , 
        "test_csv_file" 
    ]
)


