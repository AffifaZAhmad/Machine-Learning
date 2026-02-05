import pandas as pd
from scipy.sparse import csr_matrix

# Load data
ratings = pd.read_csv('../data/ratings.csv')
movies = pd.read_csv('../data/movies.csv')

# Create pivot table (movies x users)
final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
csr_data = csr_matrix(final_dataset.values)

print("Data preprocessing complete.")