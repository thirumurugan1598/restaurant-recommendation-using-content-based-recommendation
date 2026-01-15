import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("C:/Users/sarve/Downloads/Dataset .csv")
df.columns = df.columns.str.strip().str.lower()
print("Columns in dataset:", df.columns.tolist())

rename_map = {}
if 'restaurant name' in df.columns:
    rename_map['restaurant name'] = 'name'
elif 'name' in df.columns:
    rename_map['name'] = 'name'

if 'cuisines' in df.columns:
    rename_map['cuisines'] = 'cuisine'
elif 'cuisine' in df.columns:
    rename_map['cuisine'] = 'cuisine'

if 'average cost for two' in df.columns:
    rename_map['average cost for two'] = 'price'
elif 'average cost' in df.columns:
    rename_map['average cost'] = 'price'
elif 'cost' in df.columns:
    rename_map['cost'] = 'price'

if 'aggregate rating' in df.columns:
    rename_map['aggregate rating'] = 'rating'
elif 'rating' in df.columns:
    rename_map['rating'] = 'rating'

df.rename(columns=rename_map, inplace=True)

if 'cuisine' in df.columns:
    df['cuisine'] = df['cuisine'].fillna('Unknown')

if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(df['price'].median())

if 'rating' in df.columns:
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(df['rating'].mean())

if 'cuisine' in df.columns:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cuisine_encoded = encoder.fit_transform(df[['cuisine']])
    cuisine_df = pd.DataFrame(cuisine_encoded, columns=encoder.get_feature_names_out(['cuisine']))
else:
    cuisine_df = pd.DataFrame()

num_cols = []
if 'price' in df.columns:
    num_cols.append('price')
if 'rating' in df.columns:
    num_cols.append('rating')

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[num_cols]) if num_cols else np.empty((len(df),0))
for i, col in enumerate(num_cols):
    df[col + '_scaled'] = scaled[:, i]

feature_matrix = pd.concat([cuisine_df, df[[c for c in df.columns if c.endswith('_scaled')]]], axis=1)
similarity_matrix = cosine_similarity(feature_matrix)

def get_candidate_indices(cuisine_pref=None, price_min=None, price_max=None, min_rating=None):
    mask = pd.Series(True, index=df.index)
    if cuisine_pref and 'cuisine' in df.columns:
        if isinstance(cuisine_pref, (list, tuple, set)):
            mask &= df['cuisine'].isin(cuisine_pref)
        else:
            mask &= (df['cuisine'] == cuisine_pref)
    if price_min is not None and 'price' in df.columns:
        mask &= (df['price'] >= price_min)
    if price_max is not None and 'price' in df.columns:
        mask &= (df['price'] <= price_max)
    if min_rating is not None and 'rating' in df.columns:
        mask &= (df['rating'] >= min_rating)
    return df[mask].index.tolist()

def recommend_by_reference(ref_index, top_n=5, exclude_same=True):
    scores = list(enumerate(similarity_matrix[ref_index]))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    results = []
    for i, s in scores_sorted:
        if exclude_same and i == ref_index:
            continue
        results.append((i, s))
        if len(results) == top_n:
            break
    rec_indices = [i for i, _ in results]
    cols_to_show = [c for c in ['name', 'cuisine', 'price', 'rating'] if c in df.columns]
    return df.iloc[rec_indices][cols_to_show].reset_index(drop=True)

def recommend_for_user(cuisine_pref=None, price_min=None, price_max=None, min_rating=None, top_n=5):
    candidates = get_candidate_indices(cuisine_pref, price_min, price_max, min_rating)
    if not candidates:
        sort_cols = [c for c in ['rating', 'price'] if c in df.columns]
        ref_index = df.sort_values(sort_cols, ascending=[False, True]).index[0]
    else:
        ref_index = df.loc[candidates].sort_values('rating', ascending=False).index[0]
    return recommend_by_reference(ref_index, top_n=top_n, exclude_same=True)

if __name__ == "__main__":
    recommendations = recommend_for_user(
        cuisine_pref="Italian",
        price_min=250,
        price_max=500,
        min_rating=4.0,
        top_n=5
    )
    print("Top 5 recommended restaurants:")
    print(recommendations.to_string(index=False))
    recommendations.to_csv("recommended_restaurants.csv", index=False)
