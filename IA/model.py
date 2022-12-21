import numpy as np                
import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns             
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
import datetime as dt 
from datetime import datetime 
import datetime
import os
import pickle 
from collections import defaultdict
#from surprise import Reader, Dataset
#from surprise.model_selection import train_test_split, cross_validate
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
#from surprise import KNNBasic, KNNWithMeans, Dataset, accuracy, Reader, SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithZScore, BaselineOnly, CoClustering
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------------------
# DATA
# ----------------------------------------------------------

def get_data():
    articles_embeddings = pickle.load(open('articles_embeddings.pickle', 'rb'))
    df_clicks = pd.read_csv("clicks_sample.csv")
    df_articles_metadata = pd.read_csv("articles_metadata.csv", parse_dates = ["created_at_ts"] )

    return  articles_embeddings, df_clicks, df_articles_metadata


# ----------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------

def prepro(articles_embeddings, df_clicks, df_articles_metadata): 

    # delete prefix "click_" for merging tables on "article_id" later
    df_clicks = df_clicks.rename(columns = lambda x: x.strip('click_'))
    df_clicks.rename({"ountry": "country"}, axis = 1, inplace = True)
    
    # merge to get overall dataset :

    # append embeddings to df_articles_metadata
    df_articles_metadata["articles_embeddings"] = articles_embeddings.tolist()
    # Check if we get all dimensions of embeddings (250 columns,364047 rows)
    print(df_articles_metadata["articles_embeddings"].apply(len))

    df_merged = pd.merge(df_clicks, df_articles_metadata, on='article_id', how='inner')
    print("df_merged.info\n", df_merged.info)
    print("-"*30)
    print("df_merged.shape",df_merged.shape)
    print("-"*30)
    print(df_merged.head())
    print("-"*30)

    # Vérifier la présence d'autres types de valeurs manquantes non détectés par Pandas 
    print("MISSING VALUES AND DUPLICATED DATA:")
    missing_values = ["n/a", "na", "--"]
    missing = np.where(df_merged.values == missing_values)
    print(df_merged.iloc[missing])
    print("-"*30)

    # Checker nombre total de valeurs manquantes et de doublons
    print("There are", df_merged.isna().sum().sum(), "NaN")
    #print("There are", df_merged.duplicated().sum(),"Duplicated")
    print("-"*30)
    print("Check number of  users :", df_merged.user_id.value_counts()) # 707 users
    print("Check number of articles :", df_merged.article_id.nunique()) # 323 articles
    print("-"*30)

    # Delete session_size over 5 interactions
    df_dropped = df_merged[df_merged.session_size <= 5]
    print("shape after drop session_size <= 5:",df_dropped.shape)
    
    # Delete articles with <= 400 words 
    df_dropped = df_dropped[df_dropped.words_count<= 400]
    print("shape after drop words_count <= 400 :", df_dropped.shape)    
    
    return df_dropped


articles_embeddings,df_clicks, df_articles_metadata = get_data()
df_dropped = prepro(articles_embeddings, df_clicks, df_articles_metadata)

# Create a list of the best articles' rating for inserting into a carrousel's recommendation
df_best_session = df_dropped[df_dropped.session_size >= 5].sort_values(by="session_size", ascending=False)
print(df_best_session.head())

for x in df_best_session.itertuples():
    best_session_embeddings = x.articles_embeddings
    # Show only the first embedding for testing
    print(best_session_embeddings[0]) 

exit()


# ----------------------------------------------------------
# MODELS
# ----------------------------------------------------------

def training_model():

    return


