import pandas as pd
import ast


# Set display options to show all columns
pd.set_option('display.max_columns', None)  # None means no limit
# pd.set_option('display.max_rows', None)     # Optional: To view all rows if needed

import  numpy as np
movies=pd.read_csv('C:/Users/ershi/Downloads/archive (1)/tmdb_5000_movies.csv')
credits=pd.read_csv('C:/Users/ershi/Downloads/archive (1)/tmdb_5000_credits.csv')

# print(movies.columns)
# print(credits.columns)
# print(credits.head(1)['cast'])
# print(credits.head(1)['title'])

# merge dfs on the  basis of title
movies=movies.merge(credits, on='title')
mergedmovies=pd.DataFrame(movies)
# print(mergedmovies.columns)
# print(mergedmovies.info())
mergedmovies=pd.DataFrame(mergedmovies[['movie_id','title','overview', 'genres','keywords','cast','crew']])
# print(mergedmovies)
# print(mergedmovies.isnull().sum())
# mergedmovies=mergedmovies.dropna(inplace=True)

mergedmovies=pd.DataFrame(mergedmovies)
# print(mergedmovies.duplicated().sum())
print(mergedmovies.head(10))
if not mergedmovies.empty and len(mergedmovies)>0:
    genre=mergedmovies.iloc[0].genres
    print(mergedmovies.index)
    print(genre)
else:
    print("DataFrame is empty or does not have enough rows.")

def convert(obj):
    L=[]
    for i in ast.literal_eval( obj):
        L.append(i['name'])
    return L

cov=convert(genre)
# print(cov)
mergedmovies['genres']=mergedmovies['genres'].apply(convert)


mergedmovies['keywords']=mergedmovies['keywords'].apply(convert)

# cast=mergedmovies['cast'][0]
# print(cast)

def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter=counter+1
        else:
            break
    return L
# cast3=convert3(mergedmovies['cast'])
# print(cast3)
mergedmovies['cast']=mergedmovies['cast'].apply(convert3)
# print(mergedmovies['cast'])
# print(mergedmovies)
def fetch_dir(obj):
    L = []

    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])

        else:
            break
    return L

mergedmovies['crew']=mergedmovies['crew'].apply(fetch_dir)
# print(mergedmovies['crew3'])

def safe_split(x):
    if isinstance(x, str):  # Check if x is a string
        return x.split()
    else:
        return []  # Return an empty list or a placeholder for non-string values


mergedmovies['overview']=mergedmovies['overview'].apply(safe_split)
# print(mergedmovies.head(10))
mergedmovies['genres']=mergedmovies['genres'].apply(lambda x:[i.replace(" ","") for  i in x])
mergedmovies['keywords']=mergedmovies['keywords'].apply(lambda x:[i.replace(" ","") for  i in x])

mergedmovies['cast']=mergedmovies['cast'].apply(lambda x:[i.replace(" ","") for  i in x])
mergedmovies['crew']=mergedmovies['crew'].apply(lambda x:[i.replace(" ","") for  i in x])
mergedmovies['tags']=mergedmovies['overview']+mergedmovies['genres']+mergedmovies['keywords']+mergedmovies['cast']+mergedmovies['crew']
# print(mergedmovies.head(10))

new_df=mergedmovies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
print(new_df)

# print(new_df['tags'][0])
# print(new_df['tags'][1])
#text vectorization


