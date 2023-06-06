# Electronic Products Recommender System

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Usage](#usage)
4. [Scripts](#scripts)
5. [Contributing](#contributing)

## Introduction

This project is a recommender system for electronic products, built using Python. The system employs three methods of recommendation:

1. Popularity Based Filtering
2. Collaborative Filtering
3. Content-Based Filtering

The system uses a dataset of user reviews and ratings for various electronic products, and applies the three filtering techniques to generate product recommendations.

## Getting Started

These instructions will provide you with a copy of the project, which you can run on your local machine for development and testing purposes.

### Prerequisites

You will need to have the following installed on your machine:

- Python 3.8 or higher
- Jupyter Notebook

### Installation

To get a development environment running, follow these steps:

1. Clone the repo:

```bash
git clone https://github.com/cmaskaler/Product-Recommendation-Algorithms-Benchmark.git
```

2. Install required packages:
Navigate to the directory where you have cloned the repository and then install the required packages using the following command:
```bash
Copy code
pip install -r requirements.txt
```
This will install all the necessary libraries and dependencies required for this project as mentioned in the requirements.txt file.

### Usage

Once you have successfully installed all the necessary libraries and dependencies, you can run the main script by executing the following command in your terminal:
```bash
python main.py
```
This command will execute the main.py script which loads the dataset, performs some preliminary data analysis and then calls the different recommendation methods for generating product recommendations.

## Scripts Overview

This project consists of the following Python scripts:

### main.py

This is the main script that drives the entire process. It is responsible for data loading, preprocessing, and calling the methods of recommendation generation. It utilizes pandas for data handling, seaborn, and matplotlib for data visualization, and sklearn for model selection.

### popularity_filtering.py

This script contains the functionality for the Popularity Based Filtering recommendation method. It implements a basic yet powerful approach, where products are recommended based on their popularity, measured by the number of times they are interacted with (such as user ratings). The script ranks the products and recommends the top ones.

### collaborative_filtering.py

This script contains the functionality for the Collaborative Filtering recommendation method. It is based on the idea that users similar to a given user can be used to predict how much the user will like a particular product or service that those similar users have used/experienced but the user has not. The script uses user behaviour and preferences for recommending products.

### content_based_filtering.py

This script contains the functionality for the Content-Based Filtering recommendation method. This technique uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback. The script provides recommendations by comparing the content of the products.

Each script can be run independently, but for the complete recommendation system, the `main.py` should be executed.

