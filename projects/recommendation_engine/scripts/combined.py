from preprocess_data import csr_data, final_dataset, ratings, movies
from train_svd import svd
from predict_for_user import get_top_n_svd
from evaluate_models import precision_at_k_knn_existing

# Example: Recommend movies for user 503
print(get_top_n_svd(user_id=503, svd_model=svd, ratings=ratings, movies=movies))

# Evaluate KNN precision
precision_at_k_knn_existing(csr_data=csr_data, final_dataset=final_dataset, k=10)