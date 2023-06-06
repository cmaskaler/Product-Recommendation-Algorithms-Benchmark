import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import gensim
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
import webbrowser
import os
import time

# Function to load dataset
def load_dataset(dataset_path):
    print("Loading dataset...")
    start_time = time.time()
    df = pd.read_csv(dataset_path)
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    return df

# Function to create combined feature
def create_combined_feature(df):
    print("Creating combined feature 'description'...")
    start_time = time.time()
    # Combine the 'title', 'brand', and 'main_cat' columns into a single 'description' column
    df['description'] = df['title'].map(str) + " " + df['brand'].map(str) + " " + df['main_cat'].map(str)
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    return df

# Convert the text data into numerical vectors using the TF-IDF technique
def apply_tfidf(df):
    print("Applying TF-IDF Vectorizer...")
    start_time = time.time()
     # The TF-IDF Vectorizer converts text to a matrix of TF-IDF features
    tfidf_vectorizer = TfidfVectorizer()
    # We apply the vectorizer to the 'description' column of our dataframe
    # 'values.astype('U')' ensures that all values are Unicode strings
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'].values.astype('U'))
    # Calculate and print the time taken to apply TF-IDF
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    # Return the resulting TF-IDF matrix
    return tfidf_matrix

# Function to compute nearest neighbors
def compute_nearest_neighbors(reduced_matrix):
    print("Computing Nearest Neighbors...")
    start_time = time.time()
    # Define a NearestNeighbors model
    # The model uses cosine similarity as the metric, brute force to find neighbors 
    # and will find the 7 nearest neighbors of each product.
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    # Fit the model to the data
    model_knn.fit(reduced_matrix)
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    return model_knn

# Function to compute cosine similarity for a specific product
def compute_cosine_similarity(reduced_matrix, product_index):
    print("Computing Cosine Similarity for a specific product...")
    start_time = time.time()
    # Compute cosine similarity between the product at the given index and all products
    cosine_sim = cosine_similarity(reduced_matrix[product_index].reshape(1, -1), reduced_matrix)
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    return cosine_sim[0]

# Function to write the generated HTML to a file and open it in a web browser
def write_html(html, html_file):
    print("Writing HTML to file and opening in web browser...")
    start_time = time.time()
    # Open the file in write mode and write the html to it
    with open(html_file, 'w') as f:
        f.write(html)
    # Open the newly written file in the web browser
    webbrowser.open('file://' + os.path.realpath(html_file))
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")

# Function to apply Bag of Words Vectorizer
def apply_bow(df):
    print("Applying Bag of Words Vectorizer...")
    start_time = time.time()
     # The CountVectorizer converts text to a matrix of token counts
    count_vectorizer = CountVectorizer()
    # Apply the vectorizer to the 'description' column of the dataframe
    count_matrix = count_vectorizer.fit_transform(df['description'].values.astype('U'))
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    return count_matrix

# Function to apply Latent Semantic Indexing (LSI)
def apply_lsi(tfidf_matrix):
    print("Applying LSI for dimensionality reduction...")
    start_time = time.time()
    # The TruncatedSVD transformer performs linear dimensionality reduction
    lsi = TruncatedSVD(n_components=50)
    # Apply the transformer to the TF-IDF matrix
    lsi_matrix = lsi.fit_transform(tfidf_matrix)
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    return lsi_matrix

# Function to apply Latent Dirichlet Allocation (LDA)
def apply_lda(tfidf_matrix):
    print("Applying LDA for dimensionality reduction...")
    start_time = time.time()
    lda = LatentDirichletAllocation(n_components=50, random_state=0)
    lda_matrix = lda.fit_transform(tfidf_matrix)
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    return lda_matrix

# Function to apply Non-negative Matrix Factorization (NMF) for dimensionality reduction
def apply_nmf(tfidf_matrix):
    print("Applying NMF for dimensionality reduction...")
    start_time = time.time()
    nmf = NMF(n_components=50, init='random', random_state=0)
    nmf_matrix = nmf.fit_transform(tfidf_matrix)
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    return nmf_matrix

# Function to apply Principal Component Analysis (PCA)
def apply_pca(tfidf_matrix):
    print("Applying PCA for dimensionality reduction...")
    start_time = time.time()
    pca = PCA(n_components=50)
    pca_matrix = pca.fit_transform(tfidf_matrix.toarray())  # PCA requires dense vector hence conversion
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    return pca_matrix

# Function to apply Word2Vec
def apply_word2vec(df):
    print("Applying Word2Vec...")
    start_time = time.time()
    documents = [simple_preprocess(doc) for doc in df['description'].values.astype('U')]
    model_w2v = Word2Vec(documents, vector_size=50, min_count=2, workers=4)

    # Compute document vectors by averaging word vectors.
    word2vec_matrix = []
    for doc in documents:
        doc_vector = np.zeros(50)
        count = 0
        for word in doc:
            if word in model_w2v.wv:
                doc_vector += model_w2v.wv[word]
                count += 1
        if count > 0:
            doc_vector /= count
        word2vec_matrix.append(doc_vector)

    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    return np.array(word2vec_matrix)

from gensim.models import FastText

# Function to apply FastText
def apply_fasttext(df):
    print("Applying FastText...")
    start_time = time.time()
    documents = [simple_preprocess(doc) for doc in df['description'].values.astype('U')]
    model_ft = FastText(documents, vector_size=50, min_count=2, workers=4)

    # Compute document vectors by averaging word vectors.
    fasttext_matrix = []
    for doc in documents:
        doc_vector = np.zeros(50)
        count = 0
        for word in doc:
            if word in model_ft.wv:
                doc_vector += model_ft.wv[word]
                count += 1
        if count > 0:
            doc_vector /= count
        fasttext_matrix.append(doc_vector)

    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    return np.array(fasttext_matrix)


# Function to generate recommendations
def generate_recommendations(df, reduced_matrix, similarity_matrix, chosen_products, html_file, algorithm_name):
    print("Generating recommendations...")
    start_time = time.time()
    product_indices = pd.Series(df.index, index=df['id']).drop_duplicates()
    html = '<html><body><style>img {width: 200px; height: 200px} h2 {color: black;} h3 {color: red;}</style>'
    html += f'<h1>Recommendations using {algorithm_name}</h1>'

    # Iterate over chosen_products instead of random sampling
    for chosen_product in chosen_products:
        html += f'<table><tr><td><h2>{chosen_product["title"]}</h2>'
        html += f'<img src="{chosen_product["imageURL"]}"></td></tr>'

        chosen_product_index = product_indices[chosen_product["id"]]
        if isinstance(similarity_matrix, NearestNeighbors):
            distances, indices = similarity_matrix.kneighbors(reduced_matrix[chosen_product_index].reshape(1, -1))
        else:
            cosine_sim = compute_cosine_similarity(reduced_matrix, chosen_product_index)
            indices = [cosine_sim.argsort()[-6:][::-1]]

        html += '<tr>'
        counter = 0
        for i in indices[0][1:]:
            similar_product_row = df.loc[i]
            similar_product_image_url = similar_product_row['imageURL']

            # Check if the similar product has an image URL
            if pd.notna(similar_product_image_url) and counter < 5:
                similar_product_title = similar_product_row['title']
                html += f'<td><h3>{similar_product_title}</h3>'
                html += f'<img src="{similar_product_image_url}"></td>'
                counter += 1

        html += '</tr></table>'

    html += '</body></html>'
    print(f"Time taken: {round(time.time() - start_time, 2)} seconds")
    write_html(html, html_file)


# Main function to call all other functions
def initiate_recommendation():
    product_dataset_path = 'Electronics_meta.csv'
    df = load_dataset(product_dataset_path)
    df = create_combined_feature(df)

    # Select 5 random products that meet the criteria
    chosen_products = []
    while len(chosen_products) < 5:
        random_product = df.sample()
        random_product_price = pd.to_numeric(random_product['price'].str.replace('$', '', regex=False), errors='coerce').values[0]
        random_product_image_url = random_product['imageURL'].values[0]

        # Check if the product's price is not NaN, over $100 and it has an image URL
        if pd.notna(random_product_price) and random_product_price > 100.00 and pd.notna(random_product_image_url):
            chosen_products.append(random_product.iloc[0])

    # TF-IDF + SVD (LSI)
    tfidf_matrix = apply_tfidf(df)
    reduced_matrix_tfidf = apply_lsi(tfidf_matrix)
    model_knn_lsi = compute_nearest_neighbors(reduced_matrix_tfidf)
    generate_recommendations(df, reduced_matrix_tfidf, model_knn_lsi, chosen_products, 'recommendations_knn_lsi.html', 'TF-IDF + SVD (LSI)')
    generate_recommendations(df, reduced_matrix_tfidf, None, chosen_products, 'recommendations_cosine_lsi.html', 'TF-IDF + SVD (LSI) with Cosine Similarity')
    
    
    
    # BoW + SVD (LSI)
    bow_matrix = apply_bow(df)
    reduced_matrix_bow_lsi = apply_lsi(bow_matrix)
    model_knn_bow_lsi = compute_nearest_neighbors(reduced_matrix_bow_lsi)
    generate_recommendations(df, reduced_matrix_bow_lsi, model_knn_bow_lsi, chosen_products, 'recommendations_bow_lsi.html', 'BoW + SVD (LSI)')
    
    # Word2Vec
    w2v_matrix = apply_word2vec(df)
    print(w2v_matrix.shape)
    model_knn_w2v = compute_nearest_neighbors(w2v_matrix)
    print(model_knn_w2v)
    generate_recommendations(df, w2v_matrix, model_knn_w2v, chosen_products, 'recommendations_w2v.html', 'Word2Vec')
    
    # FastText
    ft_matrix = apply_fasttext(df)
    model_knn_ft = compute_nearest_neighbors(ft_matrix)
    generate_recommendations(df, ft_matrix, model_knn_ft, chosen_products, 'recommendations_ft.html', 'FastText')

