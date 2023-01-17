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
import pickle as pkl
from collections import defaultdict
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import KNNBasic, KNNWithMeans, Dataset, accuracy, Reader, SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithZScore, BaselineOnly, CoClustering
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------------------
# DATA
# ----------------------------------------------------------

def get_data():
    articles_embeddings = pkl.load(open('articles_embeddings.pickle', 'rb'))
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
    
    # append embeddings to df_articles_metadata
    df_articles_metadata["articles_embeddings"] = articles_embeddings.tolist()
    # Check if we get all dimensions of embeddings (250 columns,364047 rows)
    print(df_articles_metadata["articles_embeddings"].apply(len))
    
    # merge to get overall dataset :
    df_merged = pd.merge(df_clicks, df_articles_metadata, on='article_id', how='inner')
    print("df_merged.info\n", df_merged.info())
    print("-"*30)
    print("df_merged.shape",df_merged.shape)
    print("-"*30)
    print(df_merged.head())
    print("-"*30)

    # Check other types of NaN not detected by Pandas
    print("MISSING VALUES AND DUPLICATED DATA:")
    missing_values = ["n/a", "na", "--"]
    missing = np.where(df_merged.values == missing_values)
    print(df_merged.iloc[missing])
    print("-"*30)

    # Check total number of NaN and duplicated data
    print("There are", df_merged.isna().sum().sum(), "NaN")
    #print("There are", df_merged.duplicated().sum(),"duplicated")
    print("-"*30)
    print("Check number of  users :", df_merged.user_id.value_counts()) # 707 users
    print("Check number of articles :", df_merged.article_id.nunique()) # 323 articles
    print("-"*30)

    # Delete session_size over 5 interactions
    df_dropped = df_merged[df_merged.session_size <= 5]
    print("shape after drop session_size <= 5:", df_dropped.shape)
    
    # Delete articles with <= 400 words 
    df_dropped = df_dropped[df_dropped.words_count<= 400]
    print("shape after drop words_count <= 400 :", df_dropped.shape)
    # df_dropped.to_csv('df_dropped.csv', index = False)   
    
    return df_dropped


# ----------------------------------------------------------
# MODELS
# ----------------------------------------------------------

# We will use kNNMEANS of SURPRISE library

def training_model(df_dropped):
    """
    Args :
        df_dropped(dataframe) : dataframe cleaned after preprocessing
    Returns :
        testset : testset after split for evaluation
    """

    # Dataset's preparation for Surprise Library, we have to respect this format below:
    rdf = df_dropped[['user_id', 'article_id', 'session_size']]
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rdf, reader)
 
    # Split data in training and test sets
    trainset, testset = train_test_split(data, test_size=0.3,random_state=10)
    #print("testset" , testset)

    # Utiliser user_based true/false pour basculer entre le filtrage collaboratif basé sur l'utilisateur ou sur les articles.
    # modules de similarité : cosine , msd, pearson, pearson_baseline (shrunk)
    KNN_articles = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})

    # Add cross validation
    cross_val = cross_validate(KNN_articles, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    KNN_articles.fit(trainset)

    # Save model as a pickle object (serialization)
    filename ='KNN_articles.pkl'
    pkl.dump(KNN_articles, open(filename, 'wb'))

    return testset


def get_predictions(testset, MODEL):
    """
    Args : 
        testset : idem
        MODEL: model saved as a pickle object and loaded
    Returns :
        predictions : predictions as a list returned by Surprise
        df_reco : dataframe with predictions
        RMSE_KNN : RMSE
    """

    # test
    predictions = MODEL.test(testset)

    # Surprise returns results as a list
    print(type(predictions)); print(' ')
    #print(predictions)

    # Convert output in df for more lisibility
    df_pred = pd.DataFrame(predictions, columns = ['uid','iid','r_ui','est','details'])

    # round output of "est"
    df_pred['est'] = [np.round(x) for x in df_pred['est']]
    print(df_pred)

    # get RMSE
    RMSE_KNN = round(accuracy.rmse(predictions, verbose=True),2)
    print("RMSE_KNN",RMSE_KNN)

    # Sortir les pred par utilisateur
    df_reco = df_pred.groupby('uid').agg(article = ("iid", lambda x: list(x)))
    df_reco['pred'] = df_pred.groupby('uid').agg(pred = ("est", lambda x: list(x)))
    df_reco = df_reco.sort_values(by = "pred", ascending = False)
    print(df_reco)

    return predictions, df_reco, RMSE_KNN


# ----------------------------------------------------------
# RECOMMENDATIONS
# ----------------------------------------------------------

def get_recommendation(predictions, n=5):
    """
    Renvoie les recommendations les plus élevées pour chaque utilisateur à partir d'un ensemble de prédictions.

    Args :
        predictions(list): prédictions retournées sous forme de liste par Surprise
        n(int) : Le nombre de recommendations pour chaque utilisateur. (valeur par défaut = 5)

    Returns :
        Un dictionnaire: clés = id des utilisateurs, valeurs = listes de tuples 
        [(raw item id, rating estimation), ...] de taille n.
    """

    # Affecter les prédictions à chaque utilisateur
    top_recommendation = defaultdict(list)
    for uid, iid, r_ui, est, _ in predictions:
        top_recommendation[uid].append((iid, est))

    # Trier les prédictions pour chaque utilisateur 
    # Récupérer les k prédictions les plus élevées
    for uid, user_ratings in top_recommendation.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_recommendation[uid] = user_ratings[:n]

    # Save top_recommandation as a pickle object    
    # with open("top_recommendation.txt", "wb") as fp:
    #     pkl.dump(top_recommendation, fp)

    return top_recommendation


def find_recommendation(dico, user_id):
    results = []
    query = dico[user_id]
    for uid, est in query:
        results.append(uid)
    return results


# Get data & Preprocessing
articles_embeddings, df_clicks, df_articles_metadata = get_data()
df_dropped = prepro(articles_embeddings, df_clicks, df_articles_metadata)

# Get model: if not pickle model, train model
testset = training_model(df_dropped)
MODEL = pkl.load(open('KNN_articles.pkl', 'rb'))

# Get predictions
predictions, df_reco, RMSE_KNN = get_predictions(testset, MODEL)
print("training_model OK")

# Get recommendations
top_recommendation = get_recommendation(predictions, n=5)
print("top_recommendation OK", top_recommendation)

#Testing on user 458
user = 458
print("Reco for user :", find_recommendation(top_recommendation, user))



