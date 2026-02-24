import pandas as pd
from collections import Counter

train_path = r"artifacts/dataset/transformed_data/train_data.csv"
test_path = r"artifacts/dataset/transformed_data/test_data.csv"

def detect_columns(cols):
    u = next((c for c in cols if 'user' in c.lower() or 'reviewer' in c.lower()), None)
    i = next((c for c in cols if any(k in c.lower() for k in ['item','book','isbn','asin','product'])), None)
    r = next((c for c in cols if 'rating' in c.lower() or 'score' in c.lower()), None)
    # fallback
    if u is None and len(cols) >= 1:
        u = cols[0]
    if i is None and len(cols) >= 2:
        i = cols[1]
    return u, i, r


def summarize(df, u_col, i_col, r_col=None):
    n_inter = len(df)
    n_users = df[u_col].nunique()
    n_items = df[i_col].nunique()
    sparsity = n_inter / (n_users * n_items) if n_users and n_items else 0
    interactions_per_user = df.groupby(u_col).size()
    interactions_per_item = df.groupby(i_col).size()
    stats = {
        'interactions': n_inter,
        'users': n_users,
        'items': n_items,
        'sparsity': sparsity,
        'user_interactions_q25': int(interactions_per_user.quantile(0.25)),
        'user_interactions_median': int(interactions_per_user.median()),
        'user_interactions_q75': int(interactions_per_user.quantile(0.75)),
        'item_interactions_q25': int(interactions_per_item.quantile(0.25)),
        'item_interactions_median': int(interactions_per_item.median()),
        'item_interactions_q75': int(interactions_per_item.quantile(0.75)),
    }
    if r_col:
        stats['rating_min'] = df[r_col].min()
        stats['rating_max'] = df[r_col].max()
        stats['rating_mean'] = float(df[r_col].mean())
    return stats, interactions_per_user, interactions_per_item


if __name__ == '__main__':
    print('Loading train...')
    train = pd.read_csv(train_path, low_memory=False)
    print('Loading test...')
    test = pd.read_csv(test_path, low_memory=False)

    print('\nTrain columns:', list(train.columns))
    print('Test columns:', list(test.columns))

    u_col, i_col, r_col = detect_columns(train.columns)
    print(f'Using columns -> user: {u_col}, item: {i_col}, rating: {r_col}')

    train_stats, train_user_counts, train_item_counts = summarize(train, u_col, i_col, r_col)
    test_stats, test_user_counts, test_item_counts = summarize(test, u_col, i_col, r_col)

    print('\n--- Train summary ---')
    for k,v in train_stats.items():
        print(f'{k}: {v}')

    print('\n--- Test summary ---')
    for k,v in test_stats.items():
        print(f'{k}: {v}')

    # Cold-start checks
    train_users = set(train[u_col].unique())
    train_items = set(train[i_col].unique())
    test_users = set(test[u_col].unique())
    test_items = set(test[i_col].unique())

    cold_users = test_users - train_users
    cold_items = test_items - train_items

    print('\nCold-start in test:')
    print('test users not in train (cold users):', len(cold_users))
    print('test items not in train (cold items):', len(cold_items))

    # Fraction of test interactions that involve cold users/items
    test_cold_user_inter = test[test[u_col].isin(cold_users)].shape[0]
    test_cold_item_inter = test[test[i_col].isin(cold_items)].shape[0]
    print('test interactions with cold users:', test_cold_user_inter, '->', test_cold_user_inter / len(test))
    print('test interactions with cold items:', test_cold_item_inter, '->', test_cold_item_inter / len(test))

    # Top popular items in train and test
    top_train_items = train[i_col].value_counts().head(10).index.tolist()
    top_test_items = test[i_col].value_counts().head(10).index.tolist()
    print('\nTop 10 train items:', top_train_items)
    print('Top 10 test items:', top_test_items)

    # Basic duplicates / consistency
    dup_train = train.duplicated(subset=[u_col,i_col]).sum()
    dup_test = test.duplicated(subset=[u_col,i_col]).sum()
    print('\nDuplicate (user,item) pairs: train=', dup_train, ' test=', dup_test)

    print('\nDone.')
