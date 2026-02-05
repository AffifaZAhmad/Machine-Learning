from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pandas as pd

# Load preprocessed data
final_dataset = pd.read_csv('../data/final_dataset.csv', index_col=0)
csr_data = csr_matrix(final_dataset.values)

# Fit KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

print("KNN model ready.")