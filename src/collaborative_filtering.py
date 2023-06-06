from surprise import SVD, SVDpp, KNNBaseline, KNNBasic, KNNWithZScore, CoClustering, SlopeOne, NMF, NormalPredictor, BaselineOnly, KNNWithMeans
from surprise.model_selection import cross_validate, KFold, train_test_split
from surprise import Dataset, Reader
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# Return precision and recall at k metrics for each user
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    # First map the predictions to each user.
    user_estimated_true_ratings = defaultdict(list)
    for user_id, _, true_rating, estimated_rating, _ in predictions:
        user_estimated_true_ratings[user_id].append((estimated_rating, true_rating))

    user_precisions = dict()
    user_recalls = dict()
    for user_id, user_ratings in user_estimated_true_ratings.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        num_relevant_items = sum((true_rating >= threshold) for (_, true_rating) in user_ratings)
        # Number of recommended items in top k
        num_recommended_items_in_top_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        num_relevant_and_recommended_items_in_top_k = sum(((true_rating >= threshold) and (est >= threshold))
                              for (est, true_rating) in user_ratings[:k])
        # Precision@K: Proportion of recommended items that are relevant
        user_precisions[user_id] = num_relevant_and_recommended_items_in_top_k / num_recommended_items_in_top_k if num_recommended_items_in_top_k != 0 else 1
        # Recall@K: Proportion of relevant items that are recommended
        user_recalls[user_id] = num_relevant_and_recommended_items_in_top_k / num_relevant_items if num_relevant_items != 0 else 1

    return user_precisions, user_recalls

def perform_collaborative_filtering(ratings_df, algorithms_list, algorithm_category):
    # Sample the ratings data to reduce computation time
    ratings_sample_df = ratings_df.sample(frac=0.5)  
    print('Shape of the ratings sample :', ratings_sample_df.shape)

    # Define a cross-validation iterator
    cross_validation_split = KFold(n_splits=5)

    # Reader class is used to parse a file containing ratings
    rating_reader = Reader(rating_scale=(1, 5))

    # Load the data from the ratings_sample DataFrame into surprise's custom dataset structure
    data = Dataset.load_from_df(ratings_sample_df[['userId', 'productId', 'rating']], rating_reader)

    # Prepare a DataFrame to store the comparison results
    results_comparison_df = pd.DataFrame()

    # Process each algorithm
    for algorithm in algorithms_list:
        print(f"Processing {algorithm_category} algorithm: {algorithm}")
        
        # Perform cross validation and get measures like RMSE, MAE
        cross_validation_results = cross_validate(algorithm, data, measures=['rmse', 'mae'], cv=cross_validation_split, verbose=False)

        # Add precision and recall
        trainset, testset = train_test_split(data, test_size=.25)
        algorithm.fit(trainset)
        predictions = algorithm.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=3.5)
        cross_validation_results['precision'] = np.mean(list(precisions.values()))
        cross_validation_results['recall'] = np.mean(list(recalls.values()))

        # Get results & append algorithm name
        result_summary = pd.DataFrame.from_dict(cross_validation_results).mean(axis=0)
        result_summary = result_summary.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        result_summary = result_summary.append(pd.Series([algorithm_category], index=['Algorithm Type']))
        
        # Append to the comparison DataFrame
        results_comparison_df = results_comparison_df.append([result_summary], ignore_index=True)

    # Reset index for the comparison DataFrame
    results_comparison_df.set_index('Algorithm', inplace=True)

    # Rename columns
    results_comparison_df.columns = ['Test RMSE', 'Test MAE', 'Fit Time', 'Test Time', 'Precision', 'Recall', 'Algorithm Type']

    print(results_comparison_df)


def run_knn_for_various_k(ratings_df, k_values):
    # Sample the ratings data to reduce computation time
    ratings_sample_df = ratings_df.sample(frac=0.5)
    
    # Reader class is used to parse a file containing ratings
    rating_reader = Reader(rating_scale=(1, 5))

    # Load the data from the ratings_sample DataFrame into surprise's custom dataset structure
    data = Dataset.load_from_df(ratings_sample_df[['userId', 'productId', 'rating']], rating_reader)
    
    # Define a cross-validation iterator
    cross_validation_split = KFold(n_splits=5)

    results = []
    
    for k in k_values:
        print(f"\nProcessing KNNBaseline for k = {k}")
        
        # Initialize the KNNBaseline algorithm with the current k
        baseline_options = {'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5}
        # For the several k values we tested with faster item-based CF.
        sim_options = {'name': 'msd', 'user_based': False}
        algorithm = KNNBaseline(k=k, sim_options=sim_options, bsl_options=baseline_options)

        # Perform cross validation and get measures like RMSE
        cross_validation_results = cross_validate(algorithm, data, measures=['rmse'], cv=cross_validation_split, verbose=False)
        
        # Add precision and recall
        trainset, testset = train_test_split(data, test_size=.25)
        algorithm.fit(trainset)
        predictions = algorithm.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=3.5)
        
        # Save results
        avg_rmse = np.mean(cross_validation_results['test_rmse'])
        avg_precision = np.mean(list(precisions.values()))
        avg_recall = np.mean(list(recalls.values()))
        results.append((k, avg_rmse, avg_precision, avg_recall))

        print(f"Average RMSE for k = {k}: {avg_rmse}")
        print(f"Average Precision for k = {k}: {avg_precision}")
        print(f"Average Recall for k = {k}: {avg_recall}")

    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(*zip(*[(x[0], x[1]) for x in results]), label="RMSE")
    plt.plot(*zip(*[(x[0], x[2]) for x in results]), label="Precision")
    plt.plot(*zip(*[(x[0], x[3]) for x in results]), label="Recall")
    plt.xlabel("k")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

def init(ratings):
    # As in the thesis document, the default benchmark on CF memory-based is user-based so we set user_based=True
    # Change it for faster and better item-based CF, we tested on the experimentation section
    similarity_options = {'name': 'msd', 'user_based': True, 'verbose': False}
    baseline_options = {'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5, 'verbose': False}
    
    model_based_algorithms = [
        SVD(), 
        SVDpp(), 
        BaselineOnly(bsl_options=baseline_options), 
        CoClustering(), 
        SlopeOne(), 
        NMF()
    ]
    
    memory_based_algorithms = [
        KNNWithMeans(sim_options=similarity_options), 
        KNNBaseline(sim_options=similarity_options, bsl_options=baseline_options), 
        KNNBasic(sim_options=similarity_options), 
        KNNWithZScore(sim_options=similarity_options)
    ]

    perform_collaborative_filtering(ratings, model_based_algorithms, "Model-Based")
    perform_collaborative_filtering(ratings, memory_based_algorithms, "Memory-Based")

    k_values = list(range(5, 51, 5)) + list(range(60, 151, 10))
    run_knn_for_various_k(ratings, k_values)
