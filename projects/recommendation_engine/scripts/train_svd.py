from surprise import SVD, Dataset, Reader
import pandas as pd

# Load ratings
ratings = pd.read_csv('../data/ratings.csv')

# Prepare Surprise dataset
reader = Reader(rating_scale=(ratings.rating.min(), ratings.rating.max()))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train SVD
svd = SVD()
trainset = data.build_full_trainset()
svd.fit(trainset)

print("SVD model trained.")