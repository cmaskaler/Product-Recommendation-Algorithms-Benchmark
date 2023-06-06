import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support
import scipy.sparse as sps

def compute_popularity_scores(rating_data):
    # Calculate the sum of ratings and the number of ratings for each product
    product_grouped = rating_data.groupby(['productId']).agg({'rating': ['sum', 'count']}).reset_index()
    product_grouped.columns = ['productId', 'total_ratings', 'rating_count']

    # Standardize and scale popularity score
    product_grouped['popularity_score'] = product_grouped['total_ratings'] / product_grouped['rating_count']
    product_grouped['popularity_score'] = (product_grouped['popularity_score'] - product_grouped['popularity_score'].min()) / \
        (product_grouped['popularity_score'].max() - product_grouped['popularity_score'].min()) * 4 + 1 # Scale to 1-5

    return product_grouped

def compute_evaluation_metrics(rating_data, popularity_data):
    # Merge the ratings data with the popularity scores
    merged_data = pd.merge(rating_data, popularity_data, on='productId', how='inner')

    # Calculate RMSE and MAE
    mse_value = mean_squared_error(merged_data["rating"], merged_data["popularity_score"])
    root_mean_square_error = math.sqrt(mse_value)
    mean_absolute_error_value = mean_absolute_error(merged_data["rating"], merged_data["popularity_score"])

    print(f"RMSE: {root_mean_square_error}")
    print(f"MAE: {mean_absolute_error_value}")

    # Calculate precision and recall
    rating_threshold = rating_data['rating'].mean()
    actual_rating = merged_data["rating"] >= rating_threshold
    predicted_rating = merged_data["popularity_score"] >= rating_threshold
    precision_value, recall_value, _, _ = precision_recall_fscore_support(actual_rating, predicted_rating, average='binary')

    print(f"Precision: {precision_value}")
    print(f"Recall: {recall_value}")

    # Calculate FCP
    actual_rating_matrix = sps.coo_matrix(merged_data["rating"])
    predicted_rating_matrix = sps.coo_matrix(merged_data["popularity_score"])
    total_users = actual_rating_matrix.shape[0]
    fraction_of_concordant_pairs = np.zeros(total_users)
    for user in range(total_users):
        actual_rating_user = actual_rating_matrix.getrow(user).data
        predicted_rating_user = predicted_rating_matrix.getrow(user).data
        negative_difference_count = np.sum((actual_rating_user - predicted_rating_user) < 0)
        positive_difference_count = np.sum((actual_rating_user - predicted_rating_user) > 0)
        fraction_of_concordant_pairs[user] = (positive_difference_count - negative_difference_count) / (positive_difference_count + negative_difference_count)
    fraction_of_concordant_pairs_mean = np.mean(fraction_of_concordant_pairs)

    print(f"FCP: {fraction_of_concordant_pairs_mean}")

# Call the functions
def popularity(rating_data):
    popularity_scores = compute_popularity_scores(rating_data)
    compute_evaluation_metrics(rating_data, popularity_scores)
