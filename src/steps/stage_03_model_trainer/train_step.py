import os
import sys
import pickle
from fastapi import params
from fastapi import params
import pandas as pd 
import mlflow
from ruamel.yaml import YAML
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from transformers import data
from src.logger.log import logger
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
            # Handle both sparse CSR matrices and dense DataFrames
            if isinstance(pt, csr_matrix):
                pt_sparse = pt
            else:
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
    
    def train_popularity_baseline(self) -> None:
        """Simple baseline: recommend most popular items to all users."""
        try:
            train = pd.read_csv(self.model_trainer_config.train_csv_file)
            test = pd.read_csv(self.model_trainer_config.test_csv_file)
            
            train.drop_duplicates(subset=['userID', 'itemID'], keep='last', inplace=True)
            test.drop_duplicates(subset=['userID', 'itemID'], keep='last', inplace=True)
            
            # Get most popular items by rating count
            item_popularity = train.groupby('itemID')['rating'].count().sort_values(ascending=False)
            top_k_popular_items = item_popularity.head(TOP_K).index.tolist()
            
            # Filter test to only include users/items in training
            known_users = set(train['userID'].unique())
            test_filtered = test[test['userID'].isin(known_users)].copy()
            
            known_items = set(train['itemID'].unique())
            test_filtered = test_filtered[test_filtered['itemID'].isin(known_items)].copy()
            
            print(f"\n{'='*60}")
            print(f"POPULARITY BASELINE:")
            print(f"Top {TOP_K} popular items: {top_k_popular_items[:5]}...")
            print(f"Test set size: {len(test_filtered)}")
            
            # Create recommendations: same top-k items for all users with prediction scores
            topk_scores_formatted = []
            for user_id in test_filtered['userID'].unique():
                for rank, item_id in enumerate(top_k_popular_items):
                    # Use inverse rank as score (higher rank = lower score)
                    score = (TOP_K - rank) / float(TOP_K)
                    topk_scores_formatted.append({
                        'userID': user_id,
                        'itemID': item_id,
                        'prediction': score  # Required column for evaluation
                    })
            topk_scores = pd.DataFrame(topk_scores_formatted)
            
            eval_precision = precision_at_k(test_filtered, topk_scores, k=TOP_K, col_prediction='prediction')
            eval_recall = recall_at_k(test_filtered, topk_scores, k=TOP_K, col_prediction='prediction')
            eval_ndcg = ndcg_at_k(test_filtered, topk_scores, k=TOP_K, col_prediction='prediction')
            eval_map = map(test_filtered, topk_scores, k=TOP_K, col_prediction='prediction')
            
            print(f"Baseline Precision@{TOP_K}: {eval_precision:.4f}")
            print(f"Baseline Recall@{TOP_K}: {eval_recall:.4f}")
            print(f"Baseline nDCG@{TOP_K}: {eval_ndcg:.4f}")
            print(f"Baseline MAP@{TOP_K}: {eval_map:.4f}")
            print(f"{'='*60}\n")
            
            mlflow.set_experiment("popularity_baseline Model Experiment") 
            with mlflow.start_run():
                mlflow.log_param("model_type", "popularity_baseline")
                mlflow.log_metric("precision", eval_precision)
                mlflow.log_metric("recall", eval_recall)
                mlflow.log_metric("nDCG", eval_ndcg)
                mlflow.log_metric("MAP", eval_map)
                
        except Exception as e:
            logger.error(f"Popularity baseline failed: {e}")
            raise AppException(e, sys) from e
    
    def train_LightGCN(self) -> None :
        try:

            train = pd.read_csv(self.model_trainer_config.train_csv_file)
            test = pd.read_csv(self.model_trainer_config.test_csv_file)

            train.drop_duplicates(subset=['userID', 'itemID'], keep='last', inplace=True)
            test.drop_duplicates(subset=['userID', 'itemID'], keep='last', inplace=True)

            # Filter test to only include users and items that exist in training data
            # This prevents index out-of-bounds errors in ImplicitCF
            train_users = set(train['userID'].unique())
            train_items = set(train['itemID'].unique())
            test_before = len(test)
            test = test[(test['userID'].isin(train_users)) & (test['itemID'].isin(train_items))].copy()
            logger.info(f"Filtered test set: {test_before} -> {len(test)} pairs (removed {test_before - len(test)} invalid pairs)")

            logger.info(f'Train shape:" {train.shape} , columns:" {train.columns.tolist()}')
            logger.info(f'Train head:\n"{train.head()}')
            logger.info(f'\nTest shape:" {test.shape}, "| columns:" {test.columns.tolist()}')
            logger.info(f"Test head:\n", test.head())
            logger.info(f"\nTrain rating range:", train['rating'].min(), "-", train['rating'].max())
            logger.info(f"Test rating range:", test['rating'].min(), "-", test['rating'].max())
            logger.info(f"Unique users in train:", train['userID'].nunique(), "| in test:", test['userID'].nunique())
            logger.info(f"Unique items in train:", train['itemID'].nunique(), "| in test:", test['itemID'].nunique())


            data = ImplicitCF(
                    train=train, test=test, seed=0,
                    col_user='userID',
                    col_item='itemID',
                    col_rating='rating'
            )

            yaml_file = './src/steps/stage_03_model_trainer/recommenders_microsoft/examples/07_tutorials/KDD2020-tutorial/lightgcn.yaml'
            
            params = read_yaml_file(yaml_file)
            params['model']['epochs'] = 5 # number of epochs for training
            params['model']['embed_size'] = 64 # the embedding dimension of users and items
            params['model']['n_layers'] = 3 # number of layers of the model
            params['train']['batch_size'] = 4096  # batch size for training
            params['train']['learning_rate'] = 0.001 
            params['train']['eval_epoch'] = 1 # Evaluate every epoch
            params['train']['top_k'] = 10 # Ensure this matches the TOP_K used in evaluation
            params['train']['decay'] = 0.00001 # l2 regularization for embedding parameters
            
            # Instantiate the YAML object
            yaml_obj = YAML(typ='unsafe', pure=True) 
            yaml_obj.default_flow_style = False

            # Save YAML back to file
            with open(yaml_file , "w") as f:
                    yaml_obj.dump(params, f)

            hparams = prepare_hparams(yaml_file)
            

            model = LightGCN(hparams, data, seed=0)
            logger.info(f"{'='*20} Model Start Training. {'='*20}")
            
            mlflow.set_experiment("LightGCN Model Experiment") 

            with mlflow.start_run() : 
                logger.info(f"{model.fit()}")
                
                # Filter test data to only include users and items that were DEFINITELY in training
                test_filtered = test.drop_duplicates(subset=["userID", "itemID"]).copy()
                
                print(f"\n{'='*60}")
                print(f"TEST DATA FILTERING:")
                print(f"Original test pairs: {len(test_filtered)}")
                print(f"Training data has {len(data.user2id)} unique users and {len(data.item2id)} unique items")
                
                # Get sample of training user/item IDs to see their format
                sample_users = list(data.user2id.keys())[:5]
                sample_items = list(data.item2id.keys())[:5]
                print(f"Sample training user IDs: {sample_users}")
                print(f"Sample training item IDs: {sample_items}")
                
                # Get sample of test user/item IDs
                test_sample_users = test_filtered['userID'].head().tolist()
                test_sample_items = test_filtered['itemID'].head().tolist()
                print(f"Sample test user IDs: {test_sample_users}")
                print(f"Sample test item IDs: {test_sample_items}")
                
                # Keep only users that were in training  
                known_users = set(data.user2id.keys())
                test_filtered = test_filtered[test_filtered['userID'].isin(known_users)]
                print(f"After filtering unmapped users: {len(test_filtered)} pairs remain")
                
                # Keep only items that were in training
                known_items = set(data.item2id.keys())
                test_filtered = test_filtered[test_filtered['itemID'].isin(known_items)]
                print(f"After filtering unmapped items: {len(test_filtered)} pairs remain")
                
                print(f"Final test pairs for evaluation: {len(test_filtered)}")
                print(f"{'='*60}\n")

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

                # mlflow.log_input(train_ds, context="training")

                print(f"Model trained. metrices: Precision : {eval_precision:.4f}, Recall: {eval_recall:.4f} , nDCG: {eval_ndcg:.4f} , MAP: {eval_map:.4f}")


            #Saving model object for recommendations
            os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)
            file_name = os.path.join(self.model_trainer_config.trained_model_dir,self.model_trainer_config.trained_model_name)
            model.save(file_name)
            logger.info(f"Saving final model to {file_name}")
            
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
            logger.info(f"Saving model metadata to {metadata_file}")

        except Exception as e:
            raise AppException(e, sys) from e

    

    def initiate_model_trainer(self):
        try:
            logger.info(f"{'='*20}Model Trainer log started.{'='*20} ")
            self.train_CF()
            self.train_popularity_baseline()  # Run baseline first to establish ground truth
            self.train_LightGCN()  # Now with fixed data alignment
            logger.info(f"{'='*20}Model Trainer log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e