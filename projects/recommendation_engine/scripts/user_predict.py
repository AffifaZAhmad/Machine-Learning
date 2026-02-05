import pandas as pd

def get_top_n_svd(user_id, svd_model, ratings, movies, n=10):
    movie_ids = ratings['movieId'].unique()
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].values
    unrated_movies = [mid for mid in movie_ids if mid not in rated_movies]

    predictions = [svd_model.predict(user_id, mid) for mid in unrated_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n = predictions[:n]
    recommended_titles = []
    for pred in top_n:
        title = movies[movies['movieId'] == pred.iid]['title'].values[0]
        recommended_titles.append({'Title': title, 'Predicted Rating': round(pred.est, 2)})

    return pd.DataFrame(recommended_titles)