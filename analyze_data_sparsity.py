import pandas as pd
import numpy as np

# Read train/test
train = pd.read_csv('./artifacts/dataset/transformed_data/train_data.csv')
test = pd.read_csv('./artifacts/dataset/transformed_data/test_data.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"\nTrain unique users: {train['userID'].nunique()}, unique items: {train['itemID'].nunique()}")
print(f"Test unique users: {test['userID'].nunique()}, unique items: {test['itemID'].nunique()}")

# Sparsity
n_users = train['userID'].nunique()
n_items = train['itemID'].nunique()
sparsity = 1 - (len(train) / (n_users * n_items))
print(f"\nTrain sparsity: {sparsity:.4f} ({sparsity*100:.2f}% sparse)")
print(f"Avg interactions per user: {len(train) / n_users:.2f}")
print(f"Avg interactions per item: {len(train) / n_items:.2f}")

# Test user coverage
test_users = set(test['userID'].unique())
train_users = set(train['userID'].unique())
test_user_coverage = len(test_users & train_users) / len(test_users)
print(f"\nTest user coverage: {test_user_coverage*100:.1f}% in training")

# Test interactions per user
test_interactions = test.groupby('userID').size()
print(f"\nTest interactions per user: min={test_interactions.min()}, max={test_interactions.max()}, median={test_interactions.median():.0f}, mean={test_interactions.mean():.2f}")

# Cold start users (< 10 interactions in train)
cold_start_users = train.groupby('userID').size()
cold_start_pct = (cold_start_users < 10).sum() / len(cold_start_users)
print(f"Cold-start users in train (< 10 interactions): {cold_start_pct*100:.1f}%")

# Test users in cold-start
test_user_list = list(test_users)
train_user_interactions = train.groupby('userID').size()
test_user_cold_start = []
for u in test_user_list:
    if u in train_user_interactions.index:
        if train_user_interactions[u] < 10:
            test_user_cold_start.append(u)
print(f"Test users that are cold-start: {len(test_user_cold_start) / len(test_user_list) * 100:.1f}%")
