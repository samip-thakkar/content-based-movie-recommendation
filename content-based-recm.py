# -*- coding: utf-8 -*-
"""

@author: Samip
"""

"""Download the data set from https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip"""

#Importing the libraries
import pandas as pd


"""Pre processing"""
#Read the file to a dataframe

#Movies information
movies_df = pd.read_csv('movies.csv')
#User's information
ratings_df = pd.read_csv('ratings.csv')

#Use regular expression to find a year stored between parenthesis

#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand = False)
#Removing the parenthesis
movies_df['year'] = movies_df['year'].str.extract('(\d\d\d\d)', expand = False)
#Remove the years from title column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Apply strip to get rid of any whitespace that might have appeared
movies_df['title'] = movies_df['title'].apply(lambda x : x.strip())

#Split the genre to a list of genres
movies_df['genres'] = movies_df['genres'].str.split('|')

"""We will use One-Hot Encoding as list is not optimal in case of categorical values. 1 shows that movie has 
the genre, while 0 says it doesn't."""

#Create new dataframe
moviesWithGenres_df = movies_df.copy()

#For every row in dataframe, iterate through list of genres and place 1 in that genre column
for index, rows in movies_df.iterrows():
    for genre in rows['genres']:
        moviesWithGenres_df.at[index, genre] = 1
        
#Filling the non-genre values with 0 instead of Nan
moviesWithGenres_df = moviesWithGenres_df.fillna(0)

#Remove timestamp column from ratings as is not important
ratings_df = ratings_df.drop('timestamp', 1)


"""Content-based Recommendation System"""

#Creating a user Input
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
            ]

#Convert it to a dataframe 
inputMovies = pd.DataFrame(userInput)

#Add movieId to the user

#Filtering out movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Merging it with the title
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

#We will start with user's preferances. So lets get subset of movies that input has watched from df of genres with binary values.
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

#Reset index to avoid future issues
userMovies = userMovies.reset_index(drop = True)
#We only need genre table, so drop other columns.
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

"""we're going to turn each genre into weights. We can do this by using the input's reviews and multiplying them 
into the input's genre table and then summing up the resulting table by column. This operation is actually a 
dot product between a matrix and a vector, so we can simply accomplish by calling Pandas's "dot" function."""

#DotProduct to get the weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

#Get genre of every movie in original DataFrame
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#Drop theunnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

#Multiply genres by weights and then take the weighted average
recommendationTable_df = ((genreTable * userProfile).sum(axis = 1)) / (userProfile.sum())

#Sort recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending = False)

#Final movie recommendation table
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())])