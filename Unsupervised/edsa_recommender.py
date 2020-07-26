
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

#Functions
def explore_data(dataset):
        df = pd.read_csv(os.path.join(dataset))
        return df

# App declaration
def main():

    page_options = ["Recommender System","Solution Overview", "Search Your Movie","About App"]


    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")
        st.header("Exploratory Data Analysis")
        st.subheader("Dataset")


        dataset_ratings = 'ratings.csv'


        if st.checkbox("Preview DataFrame"):
            data = explore_data('resources/data/movies.csv')
            if st.button("Head"):
                st.write(data.head())
            if st.button("Tail"):
                st.write(data.tail())


        # Dimensions
        if st.checkbox("Show Dimensions"):
            if st.button("Rows"):
                data = explore_data('resources/data/movies.csv')
                st.text("Length of Rows")
                st.write(len(data))
            if st.button("Columns"):
                data = explore_data('resources/data/movies.csv')
                st.text("Length of Columns")
                st.write(data.shape[1])


        # Bar Plot
        if st.checkbox("Bar Plot"):
            if st.button("Ratings"):
                data = explore_data('resources/data/ratings.csv')
                v_counts = data.rating.value_counts()
                st.bar_chart(v_counts)
            if st.button("Genres"):
                data = explore_data('resources/data/movies.csv')
                words = data['genres'].apply(lambda x : x.split("|"))
                words_list = words.tolist()
                flat_list = []
                for sublist in words_list:
                    for item in sublist:
                        flat_list.append(item)
                dict_ = Counter(flat_list)
                dict_ = sorted(dict_.items(), key=lambda x: x[1], reverse=True)
                x_val = [x[0] for x in dict_]
                y_val = [x[1] for x in dict_]

                plt.barh(x_val,y_val)
                plt.show()
                st.pyplot()

        # Plot Hist
        if st.checkbox("Histogram"):
            hist_values = np.histogram(data[''].dt.hour, bins=24, range=(0,24))[0]
            st.bar_chart(hist_values)

        # Top 10 Movies per User
        if st.checkbox("Highest Rated Movies Per User"):
            train = explore_data('resources/data/train.csv')
            movies = explore_data('resources/data/movies.csv')
            user_input = st.number_input("Insert UserId")
            if user_input:
                merged = pd.merge(train,movies, on = 'movieId')
                cols = merged.columns.tolist()
                cols = cols[-2:] + cols[:-2]
                merged = merged[cols]
                df = merged[merged['userId'] == user_input].sort_values(by = 'rating', ascending = False)
                st.write(df.head(10))

    # About
    if page_selection == "About App":
        st.subheader("Movie Recommender")
        st.text("Built with Streamlit")
        st.text('Group SY5')

    if page_selection == "Search Your Movie":
        data = explore_data('resources/data/movies.csv')
        if st.button('Based On Title'):
            user_input = st.text_input("Movie Title", "Toy Story (1995)")
            df = data[data['title'] == str(user_input)]
            st.write(df)
        if st.button('Based On Genre'):
            words = data['genres'].apply(lambda x : x.split("|"))
            data['new'] = words
            data['new'] = data['new'].apply(lambda x: ' '.join([str(elem) for elem in x]))
            user_input = st.text_input("Movie Genre")
            if user_input:
                df1 = data[data['new'].str.contains(user_input)]
                st.write(df1)



if __name__ == '__main__':
    main()
