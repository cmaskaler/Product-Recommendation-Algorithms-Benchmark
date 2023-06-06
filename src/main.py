# Utilities
import warnings
from IPython.core.interactiveshell import InteractiveShell

# Mathematical calculation
from sklearn import model_selection

# Data handling
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import the custom modules
import popularity_filtering as popularity_filtering
import collaborative_filtering as collaborative_filtering
import content_based_filtering as content_based_filtering

# Configure for any default setting of any library
InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load the dataset into a Pandas dataframe called ratings and skip any lines that return an error
ratings = pd.read_csv('data\Electronics.csv',
                      names=['userId', 'productId', 'rating', 'timestamp'],
                      error_bad_lines=False,
                      warn_bad_lines=False)

# Print the information about the loaded dataset
print("Opened 'Electronics.csv' with columns:")
print(list(ratings.columns))

# Save an original copy of the dataframe
ratings_original = ratings.copy(deep=True)
# Check the head of the dataset
ratings.head()
# Check the tail of the dataset
ratings.tail()

# Get the shape and size of the dataset
print("Number of reviews    :",ratings.shape[0])

#Dropping the Timestamp column
ratings.drop(['timestamp'], axis=1,inplace=True)

# Check the count of unique user and product data
unique_original = (ratings.userId.nunique(), ratings.productId.nunique())
print('Count of unique Users    :', unique_original[0])
print('Count of unique Products :', unique_original[1])

# Check the distribution of ratings 
print('Count of reviews for each rating:')
print(ratings.rating.value_counts()) # Explicitly print the output of value_counts()

# Use Seaborn's catplot function to plot the count of each rating
g = sns.catplot(x="rating", data=ratings, aspect=2.0, kind='count')
g.set_ylabels("Total number of ratings")
plt.ticklabel_format(style='plain', axis='y')

# Get the count values for each rating
rating_counts = ratings.rating.value_counts().sort_index()
for i, count in enumerate(rating_counts):
    g.ax.text(i, count, str(count), ha='center', va='bottom')

# Show the plot
plt.show()


# Find the unique products under each ratings
ratings.groupby('rating')['productId'].nunique()

# Find the density of the rating matrix
print('Total observed ratings in the dataset  :', len(ratings))
possible_num_of_ratings = ratings.userId.nunique() * ratings.productId.nunique()
print('Total ratings possible for the dataset :', possible_num_of_ratings)
density = len(ratings) / possible_num_of_ratings * 100
print('Density of the dataset                 : {:4.5f}%'.format(density))

# Count the number of ratings for each product
prodID = ratings.groupby('productId')['rating'].count()

# Find products that have 100 or more ratings
# Change to 150 or 200 if you encounter memory issues
top_prod = prodID[prodID >= 100].index

# Keep data only for products that have 50 or more ratings
ratings = ratings[ratings['productId'].isin(top_prod)]

print('Unique users who have rated 1 or more products :', ratings.userId.nunique())
print('\nFinal length of the dataset :', len(ratings))

# Find the density of the final matrix
final_ratings_matrix = ratings.pivot_table(index='userId', columns='productId', values='rating', fill_value=0)

# Converting to sparse dataframe
final_ratings_matrix = final_ratings_matrix.astype(pd.SparseDtype("float", 0))

print('Shape of final_ratings_matrix:', final_ratings_matrix.shape)


print('Total observed ratings in the dataset  :', len(ratings))
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('Total ratings possible for the dataset :', possible_num_of_ratings)
density = len(ratings) / possible_num_of_ratings * 100
print('Density of the dataset                 : {:4.2f}%'.format(density))

# Call the popularity function from the popularity_filtering module
popularity_filtering.popularity(ratings)
collaborative_filtering.init(ratings)
content_based_filtering.initiate_recommendation()
