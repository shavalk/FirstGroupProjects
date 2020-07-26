
# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Creating needed data
    data = data_preprocessing(27000)
    titles = data['title']
    indices = pd.Series(data.index, index=data['title'])
    # Instantiating and generating the count matrix
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(data['keyWords'])
    title1 = movie_list[0]
    title2 = movie_list[1]
    title3 = movie_list[2]
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # Getting the index of the movie that matches the title
    idx_1 = indices[title1]
    idx_2 = indices[title2]
    idx_3 = indices[title3]
    # Creating a Series with the similarity scores in descending order
    sim_scores1 = list(enumerate(cosine_sim[idx_1]))
    sim_scores1 = sorted(sim_scores1, key=lambda x: x[1], reverse=True)
    sim_scores2 = list(enumerate(cosine_sim[idx_2]))
    sim_scores2 = sorted(sim_scores2, key=lambda x: x[1], reverse=True)
    sim_scores3 = list(enumerate(cosine_sim[idx_3]))
    sim_scores3 = sorted(sim_scores3, key=lambda x: x[1], reverse=True)
    sim_scores = np.concatenate((sim_scores1,sim_scores2))
    sim_scores_final = np.concatenate((sim_scores, sim_scores3))
    sim_scores_f = np.sort(sim_scores_final)[::-1]
    sim_scores_f = sim_scores_final[:30]
    # Store movie indices
    movie_indices = [i[0] for i in sim_scores_f]
    res = []
    [res.append(x) for x in movie_indices if x not in res]
    res = res[:top_n]
    # Convert the indexes back into titles
    return titles.iloc[res]
