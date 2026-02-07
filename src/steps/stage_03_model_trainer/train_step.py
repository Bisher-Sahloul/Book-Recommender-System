import os
import sys
import pickle
import pandas as pd 
import mlflow
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from src.logger.log import logging
from src.config.configuration import AppConfiguration
from src.exception.exception_handler import AppException
from src.utils.util import clone_github_repo
from src.constant import * 

import pandas as pd
import numpy as np
import tensorflow as tf
from src.utils.util import read_yaml_file
from src.steps.stage_03_model_trainer.recommenders_microsoft.recommenders.utils.timer import Timer
from src.steps.stage_03_model_trainer.recommenders_microsoft.recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from src.steps.stage_03_model_trainer.recommenders_microsoft.recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from src.steps.stage_03_model_trainer.recommenders_microsoft.recommenders.datasets.python_splitters import python_stratified_split
from src.steps.stage_03_model_trainer.recommenders_microsoft.recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from src.steps.stage_03_model_trainer.recommenders_microsoft.recommenders.utils.constants import SEED as DEFAULT_SEED
from src.steps.stage_03_model_trainer.recommenders_microsoft.recommenders.models.deeprec.deeprec_utils import prepare_hparams
from src.steps.stage_03_model_trainer.recommenders_microsoft.recommenders.utils.notebook_utils import store_metadata


class ModelTrainer:
    def __init__(self, app_config = AppConfiguration()):
        try:
            self.model_trainer_config = app_config.get_model_trainer_config()
            self.data_validation_config = app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e
                
    def train_CF(self) -> None : 
        try : 
            file_path = os.path.join(
                        self.data_validation_config.serialized_objects_dir,
                        "piovt_table_data.pkl"
            )
            with open(file_path, "rb") as f:
                pt = pickle.load(f)            
            pt_sparse = csr_matrix(pt.values)
            item_similarity = cosine_similarity(pt_sparse.T, dense_output=False)
            file_path = os.path.join(
                self.data_validation_config.serialized_objects_dir,
                "item_similarity.pkl"
            )
            with open(file_path , "wb") as f:
                pickle.dump(item_similarity, f)

        except Exception as e : 
            raise AppException(e , sys) from e 
    
    def train_LightGCN(self) -> None :
        try:

            train = pd.read_csv(self.model_trainer_config.train_csv_file)
            test = pd.read_csv(self.model_trainer_config.test_csv_file)


            logging.info(f'Train shape:" {train.shape} , columns:" {train.columns.tolist()}')
            logging.info(f'Train head:\n"{train.head()}')
            logging.info(f'\nTest shape:" {test.shape}, "| columns:" {test.columns.tolist()}')
            logging.info(f"Test head:\n", test.head())
            logging.info(f"\nTrain rating range:", train['rating'].min(), "-", train['rating'].max())
            logging.info(f"Test rating range:", test['rating'].min(), "-", test['rating'].max())
            logging.info(f"Unique users in train:", train['userID'].nunique(), "| in test:", test['userID'].nunique())
            logging.info(f"Unique items in train:", train['itemID'].nunique(), "| in test:", test['itemID'].nunique())


            data = ImplicitCF(
                    train=train, test=None, seed=0,
                    col_user='userID',
                    col_item='itemID',
                    col_rating='rating'
            )

            yaml_file = './src/steps/stage_03_model_trainer/recommenders_microsoft/examples/07_tutorials/KDD2020-tutorial/lightgcn.yaml'

            hparams = prepare_hparams(yaml_file)

            params = read_yaml_file(yaml_file)

            model = LightGCN(hparams, data, seed=0)
            logging.info(f"{'='*20} Model Start Training. {'='*20}")

            with mlflow.start_run() : 
                logging.info(f"{model.fit()}")
                # Filter test users to those present in the training `user2id` mapping
                if hasattr(data, 'user2id') and isinstance(data.user2id, dict):
                    known_user_mask = test['userID'].isin(list(data.user2id.keys()))
                    unknown_count = (~known_user_mask).sum()
                    if unknown_count > 0:
                        logging.info(f"Found {unknown_count} users in test not seen during training - skipping them for evaluation.")
                    test_filtered = test[known_user_mask].copy()
                else:
                    test_filtered = test

                topk_scores = model.recommend_k_items(test_filtered, top_k = TOP_K)
                eval_map = map(test_filtered, topk_scores, k=TOP_K)
                eval_ndcg = ndcg_at_k(test_filtered, topk_scores, k=TOP_K)
                eval_precision = precision_at_k(test_filtered, topk_scores, k=TOP_K)
                eval_recall = recall_at_k(test_filtered, topk_scores, k=TOP_K)
                # Log params, metrics, and model

                mlflow.log_param("model_type", params['model']['model_type'])
                mlflow.log_param("embed_size", params['model']['embed_size'])
                mlflow.log_param("n_layers" , params['model']['n_layers'])
                
                mlflow.log_param("batch_size" , params["train"]["batch_size"])
                mlflow.log_param("decay" , params["train"]["decay"])
                mlflow.log_param("epochs" , params["train"]["epochs"])
                mlflow.log_param("learning_rate" , params["train"]["learning_rate"])

                mlflow.log_metric("precision", eval_precision)
                mlflow.log_metric("recall", eval_recall)
                mlflow.log_metric("nDCG", eval_ndcg)
                mlflow.log_metric("MAP", eval_map)

                model_dir = os.path.join(self.data_validation_config.serialized_objects_dir, "lightgcn_model")
                model.save(model_dir)

                mlflow.log_artifacts(model_dir, artifact_path="LightGCN_model")

                # train_ds = mlflow.data.from_pandas(train , source="training_data")
                # mlflow.log_input(train_ds, context="training")

                print(f"Model trained. metrices: Precision : {eval_precision:.4f}, Recall: {eval_recall:.4f} , nDCG: {eval_ndcg:.4f} , MAP: {eval_map:.4f}")


            #Saving model object for recommendations
            os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)
            file_name = os.path.join(self.model_trainer_config.trained_model_dir,self.model_trainer_config.trained_model_name)
            model.save(file_name)
            logging.info(f"Saving final model to {file_name}")
            
            # Save model metadata and inference data (cannot pickle TensorFlow objects)
            metadata = {
                'user2id': data.user2id,
                'item2id': data.item2id,
                'id2user': data.id2user,
                'id2item': data.id2item,
                'n_users': data.n_users,
                'n_items': data.n_items,
                'R': data.R,  # User-item interaction matrix
                'col_user': data.col_user,
                'col_item': data.col_item,
                'col_rating': data.col_rating,
                'batch_size': model.batch_size  # Needed for scoring
            }
            metadata_file = os.path.join(self.model_trainer_config.trained_model_dir, 'model_metadata.pkl')
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            logging.info(f"Saving model metadata to {metadata_file}")

        except Exception as e:
            raise AppException(e, sys) from e

    

    def initiate_model_trainer(self):
        try:
            logging.info(f"{'='*20}Model Trainer log started.{'='*20} ")
            self.train_CF()
            self.train_LightGCN()
            logging.info(f"{'='*20}Model Trainer log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e