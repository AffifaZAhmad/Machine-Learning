**MOVIE RECOMMENDATION SYSTEM:**

This project builds a **Movie Recommendation System** using **Collaborative Filtering** techniques. It uses both **K-Nearest Neighbors (KNN)** with **Cosine Similarity for item based recommendation** and **Matrix Factorization (SVD) for user based recommendation** for improved rating predictions and evaluation.

### **Technologies Used**

* Python (Jupyter Notebook)

* Pandas, NumPy

* SciPy (CSR Matrix)

* Scikit-learn (KNN)

* Surprise (for SVD & RMSE)

* Gradio (for GUI-based movie search)  
  


### **Recommendation Techniques**

#### **1\. KNN with Cosine Similarity**

* A user-item matrix was created from ratings.

* Cosine similarity was calculated between movies to find nearest neighbors.

* Recommended movies are based on the closest movies (K=10) in vector space.

#### **2\. SVD (Surprise library)**

* Decomposes the rating matrix into latent factors.

* Predicts missing ratings and recommends top-N movies for each user.

* Can be evaluated with RMSE or Precision@K.

### 

###  **Evaluation**

####  **KNN Evaluation:**

* **Precision@10** was calculated to measure the proportion of relevant recommendations among the top 10\.

* Higher precision indicates more relevant movie suggestions.

####  **SVD Evaluation (optional):**

* **Root Mean Squared Error (RMSE)** used to evaluate predicted ratings against real ratings.

* **Precision@K** can also be computed for top-N ranking accuracy.

### 

### **How to Use**

* Use the Gradio GUI to type a movie name (e.g., `Avatar`) and click on KNN.

* The system returns the **10 most similar movies** based on cosine similarity.

* Use the Gradio GUI to type a user id (e.g., `501`) and click on SVD.

* The system returns the **10 most similar movies** based on the user 501 and his top rated choices.

