import os
import sys
import pandas as pd 
import numpy as np 
from fastapi import FastAPI , Request , Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from typing import Optional
from typing import List 
from api.vectordb import search_books
from api.models import Book
from api.database import BOOKS 
from src.config.configuration import AppConfiguration
import pickle
from scipy.sparse import csr_matrix
import tensorflow as tf
import numpy as np

# Add the recommenders_microsoft path to sys.path to resolve module imports
recommenders_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'steps', 'stage_03_model_trainer', 'recommenders_microsoft')
sys.path.insert(0, recommenders_path)

from src.steps.stage_03_model_trainer.recommenders_microsoft.recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from src.steps.stage_03_model_trainer.recommenders_microsoft.recommenders.utils.python_utils import get_top_k_scored_items

# Global cache for model
_model_cache = {
    'model': None,
    'metadata': None,
    'loaded': False
}

class LightGCNInferenceWrapper:
    """Wrapper to enable inference on a loaded LightGCN model with metadata."""
    def __init__(self, sess, metadata):
        self.sess = sess
        self.data = type('obj', (object,), {
            'user2id': metadata['user2id'],
            'item2id': metadata['item2id'],
            'id2user': metadata['id2user'],
            'id2item': metadata['id2item'],
            'n_users': metadata['n_users'],
            'n_items': metadata['n_items'],
            'R': metadata['R'],
            'col_user': metadata['col_user'],
            'col_item': metadata['col_item'],
            'col_rating': metadata['col_rating']
        })()
        self.batch_size = metadata['batch_size']
        
        # Discover tensor names from the loaded graph
        self._discover_tensors()
        
    def _discover_tensors(self):
        """Discover placeholder and output tensor names from the loaded graph."""
        graph = self.sess.graph
        
        # Find placeholder operations
        placeholders = [op for op in graph.get_operations() if op.type == 'Placeholder']
        self.placeholder_list = [op.outputs[0] for op in placeholders]
        
        # Print discovered tensors for debugging
        print(f"Found {len(self.placeholder_list)} placeholders: {[op.name for op in placeholders]}")
        
        # For generic placeholders, assume order: users, pos_items, neg_items (neg_items for training only)
        # We need at least 2 placeholders for inference
        self.users_placeholder = self.placeholder_list[0] if len(self.placeholder_list) > 0 else None
        self.pos_items_placeholder = self.placeholder_list[1] if len(self.placeholder_list) > 1 else None
        
        # Find batch_ratings - usually a MatMul operation
        self.batch_ratings_tensor = None
        matmul_ops = [op for op in graph.get_operations() if op.type == 'MatMul']
        
        if matmul_ops:
            # Use the last MatMul as batch_ratings
            self.batch_ratings_tensor = matmul_ops[-1].outputs[0]
        
        users_name = self.users_placeholder.name if self.users_placeholder is not None else 'Not found'
        items_name = self.pos_items_placeholder.name if self.pos_items_placeholder is not None else 'Not found'
        ratings_name = self.batch_ratings_tensor.name if self.batch_ratings_tensor is not None else 'Not found'
        
        print(f"Users placeholder: {users_name}")
        print(f"Pos items placeholder: {items_name}")
        print(f"Batch ratings tensor: {ratings_name}")
        
    def score(self, user_ids, remove_seen=True):
        """Score all items for given users."""
        try:
            if self.users_placeholder is None or self.pos_items_placeholder is None or self.batch_ratings_tensor is None:
                raise Exception("Could not find required tensors in the model graph")
            
            test_scores = []
            u_batch_size = self.batch_size
            n_user_batchs = len(user_ids) // u_batch_size + 1
            
            for u_batch_id in range(n_user_batchs):
                start = u_batch_id * u_batch_size
                end = (u_batch_id + 1) * u_batch_size
                user_batch = user_ids[start:end]
                item_batch = np.arange(self.data.n_items)
                
                rate_batch = self.sess.run(
                    self.batch_ratings_tensor, 
                    {self.users_placeholder: user_batch, self.pos_items_placeholder: item_batch}
                )
                test_scores.append(np.array(rate_batch))
            
            test_scores = np.concatenate(test_scores, axis=0)
            
            if remove_seen:
                test_scores += self.data.R.tocsr()[user_ids, :] * -np.inf
            
            return test_scores
        except Exception as e:
            print(f"Error in score: {e}")
            raise
    
    def recommend_k_items(self, user_ids, top_k=5, remove_seen=True):
        """Recommend top K items for users."""
        try:
            scores = self.score(user_ids, remove_seen=remove_seen)
            top_items, top_scores = get_top_k_scored_items(scores, top_k=top_k, sort_top_k=True)
            
            # Convert item IDs back to original item names
            recommendations = []
            for item_ids in top_items:
                items = [self.data.id2item[int(item_id)] for item_id in item_ids]
                recommendations.append(items)
            
            return recommendations
        except Exception as e:
            print(f"Error in recommend_k_items: {e}")
            raise

def load_lightgcn_model_and_metadata():
    """Load LightGCN model weights and metadata from disk."""
    global _model_cache
    
    if _model_cache['loaded']:
        return _model_cache['model']
    
    try:
        model_dir = AppConfiguration().get_model_trainer_config().trained_model_dir
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load model checkpoint
        model_path = os.path.join(model_dir, 'model')  # TensorFlow checkpoint path
        
        # Create TensorFlow session to restore model
        tf.compat.v1.reset_default_graph()
        
        # Import the saved graph structure from the meta file
        meta_graph_path = model_path + '.meta'
        saver = tf.compat.v1.train.import_meta_graph(meta_graph_path)
        
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        
        # Restore the variables
        saver.restore(sess, model_path)
        
        # Create inference wrapper
        model_wrapper = LightGCNInferenceWrapper(sess, metadata)
        
        _model_cache['model'] = model_wrapper
        _model_cache['metadata'] = metadata
        _model_cache['loaded'] = True
        
        print(f"✓ LightGCN model loaded from {model_dir}")
        return model_wrapper
        
    except Exception as e:
        print(f"✗ Error loading LightGCN model: {e}")
        import traceback
        traceback.print_exc()
        return None


app = FastAPI(title="Book Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def tansform_to_dict(data: pd.DataFrame):
    rows = []
    for _, row in data.iterrows():
        rows.append({
                   "ISBN": row["ISBN"],
                    "Book_Title": row["Book-Title"],
                    "Book_Author": row["Book-Author"],
                    "Year_Of_Publication": row["Year-Of-Publication"],
                    "Publisher": row["Publisher"] ,
                    "Description" : row["Description"] , 
                    "Categories" : row["Categories"] , 
                    "Image" : row["Image"] , 
                    "rating" : row["rating"]
        })
    return rows

def item_based_recommend(
    ISBN,
    pt_sparse,
    pt,
    item_similarity,
    top_n_items = 5
):
    # index of the book
    item_idx = pt.columns.get_loc(ISBN)

    # book vector (1 × n_items)
    item_vector = item_similarity[item_idx]
         
    # similarity scores
    scores = item_vector.toarray().flatten()

    # remove the book itself
    scores[item_idx] = 0

    # top-N similar books
    top_items_idx = np.argsort(scores)[-top_n_items:][::-1]

    return pt.columns[top_items_idx]




app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_root():
    return RedirectResponse(url="/static/app.html")


# 🔍 Vector search
@app.get("/api/search", response_model = List[Book])
def search(q: str = Query(..., min_length=2)):
    return search_books(q)

# ⭐ Popular books
@app.get("/api/books/popular", response_model=List[Book])
def popular_books():

    path = os.path.join(AppConfiguration().get_data_transformation_config().transformed_data_dir , 'most_popular_books.pkl')

    most_popular_books = pickle.load(open(path, 'rb'))

    return tansform_to_dict(most_popular_books[:5])

# 👤 User recommendations
@app.get("/api/user/recommendations", response_model = List[Book])
def user_recommendations(user_id: str = "A2ZE4PQJ4TR0CH"):
    """
    Get personalized book recommendations for a user using LightGCN model.
    Returns top 5 recommended books based on collaborative filtering.
    Falls back to popular books if model loading fails.
    """
    try:
        model_wrapper = load_lightgcn_model_and_metadata()
        
        if model_wrapper is None:
            raise Exception("Failed to load model")
        
        # Check if user exists in training data
        user2id = model_wrapper.data.user2id
        
        if user_id not in user2id:
            print(f"User {user_id} not found in training data. Returning popular books.")
            # Fallback to popular books
            path = os.path.join(AppConfiguration().get_data_transformation_config().transformed_data_dir, 'most_popular_books.pkl')
            most_popular_books = pickle.load(open(path, 'rb'))
            return tansform_to_dict(most_popular_books[:5])
        
        # Get user's internal ID
        user_internal_id = user2id[user_id]
        user_ids = np.array([user_internal_id])

        # Get recommendations
        recommendations = model_wrapper.recommend_k_items(user_ids, top_k = 10)
        recommended_item_ids = recommendations[0]  # Get first user's recommendations

        if len(recommended_item_ids) > 5 : 
            recommended_item_ids = recommended_item_ids[:5]
        
        #print(f"Recommended items for user {user_id}: {recommended_item_ids}")
        
        # Convert item IDs to ISBNs and get book details
        recommended_books = BOOKS[BOOKS["ISBN"].isin(recommended_item_ids)]


        return tansform_to_dict(recommended_books) if len(recommended_books) > 0 else []
        
    except Exception as e:
        print(f"Error in user recommendations: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to popular books if anything fails
        try:
            path = os.path.join(AppConfiguration().get_data_transformation_config().transformed_data_dir, 'most_popular_books.pkl')
            most_popular_books = pickle.load(open(path, 'rb'))
            return tansform_to_dict(most_popular_books[:5])
        except:
            return []

# 📖 Book detail
@app.get("/api/books/{book_id}", response_model=Book)
def get_book(book_id: str):
    for _,row in BOOKS.iterrows():
        if row["ISBN"] == book_id:
            return row.to_dict()
    return None

# 🔁 Related books
@app.get("/api/books/{book_id}/related", response_model=List[Book])
def related_books(book_id: str):

    path =  AppConfiguration().get_data_validation_config().serialized_objects_dir

    item_similarity = pickle.load(open(os.path.join(path , "item_similarity.pkl") , 'rb'))
    
    pt = pickle.load(open(os.path.join(path , "piovt_table_data.pkl" ) , 'rb'))
    
    pt_sparse = csr_matrix(pt.values)
    
    recommended_ISBNS = item_based_recommend(
        ISBN=book_id,
        pt_sparse=pt_sparse,
        pt=pt,
        item_similarity=item_similarity,
        top_n_items=5
    )

    return tansform_to_dict(BOOKS[BOOKS["ISBN"].isin(recommended_ISBNS)])