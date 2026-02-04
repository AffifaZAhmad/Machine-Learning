from sklearn.neighbors import NearestNeighbors
import pandas as pd

def precision_at_k_knn_existing(csr_data, final_dataset, k=10):
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k+1, n_jobs=-1)
    knn.fit(csr_data)

    hits, total = 0, 0
    movie_ids = final_dataset.index.tolist()

    for i in range(len(movie_ids)):
        movie_id = movie_ids[i]
        distances, indices = knn.kneighbors(csr_data[i], n_neighbors=k+1)
        neighbor_ids = [movie_ids[idx] for idx in indices.squeeze().tolist()[1:]]

        actual_users = set(final_dataset.loc[movie_id][final_dataset.loc[movie_id] > 0].index)

        for neighbor_id in neighbor_ids:
            neighbor_users = set(final_dataset.loc[neighbor_id][final_dataset.loc[neighbor_id] > 0].index)
            hits += len(actual_users.intersection(neighbor_users))
            total += len(neighbor_users)

    precision = round(hits / total, 4) if total > 0 else 0
    print(f"Precision@{k} for KNN:", precision)