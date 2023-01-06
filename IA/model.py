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
    print("shape after drop session_size <= 5:",df_dropped.shape)
    
    # Delete articles with <= 400 words 
    df_dropped = df_dropped[df_dropped.words_count<= 400]
    print("shape after drop words_count <= 400 :", df_dropped.shape)    
    
    return df_dropped


articles_embeddings,df_clicks, df_articles_metadata = get_data()
df_dropped = prepro(articles_embeddings, df_clicks, df_articles_metadata)

# Create a list of the best articles' rating for inserting into a carrousel's recommendation
#df_best_session = df_dropped[df_dropped.session_size >= 5].sort_values(by="session_size", ascending=False)
#print(df_best_session.head())

#for x in df_best_session.itertuples():
    #best_session_embeddings = x.articles_embeddings
    # Show only the first embedding for testing
    #print(best_session_embeddings[0]) 

#exit()


# ----------------------------------------------------------
# MODELS
# ----------------------------------------------------------

# We will use kNNMEANS of SURPRISE library

def training_model(df_dropped):
    """
    Args :
        df_dropped(dataframe) : dataframe cleaned after preprocessing
    Returns :
        predictions : predictions as a list returned by Surprise
        df_reco : dataframe with predictions
        MODEL : model saved as a pickle object
        RMSE_KNN : RMSE
    """

    # Dataset's preparation for Surprise Library, we have to respect this format below:
    rdf = df_dropped[['user_id', 'article_id', 'session_size']]
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rdf, reader)
 
    # Split data in training and test sets
    trainset, testset = train_test_split(data, test_size=0.3,random_state=10)
    print(testset)

    # Utiliser user_based true/false pour basculer entre le filtrage collaboratif basé sur l'utilisateur ou sur les articles.
    # modules de similarité : cosine , msd, pearson, pearson_baseline (shrunk)
    KNN_articles = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})

    # Add cross validation
    cross_val = cross_validate(KNN_articles, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    KNN_articles.fit(trainset)

    # test
    predictions = KNN_articles.test(testset)

    # Surprise returns results as a list
    print(type(predictions)); print(' ')
    print(predictions)

    # Convert output in df for more lisibility
    df_pred = pd.DataFrame(predictions, columns = ['uid','iid','r_ui','est','details'])

    # round output of "est"
    df_pred['est'] = [np.round(x) for x in df_pred['est']]
    print(df_pred)

    # get RMSE
    RMSE_KNN = round(accuracy.rmse(predictions, verbose=True),2)
    print(RMSE_KNN)

    # tester sur l'utilisateur 458, [3] => 'est'
    #KNN_articles.predict(uid = 458, iid = 236682)[3]

    # Sortir les pred par utilisateur
    df_reco = df_pred.groupby('uid').agg(article = ("iid", lambda x: list(x)))
    df_reco['pred'] = df_pred.groupby('uid').agg(pred = ("est", lambda x: list(x)))
    df_reco = df_reco.sort_values(by = "pred", ascending = False)
    print(df_reco)

    # Enregistrer le modèle au format pickle (sérialisation)
    filename ='KNN_articles.pkl'
    MODEL = pickle.dump(KNN_articles, open(filename, 'wb'))

    return predictions, df_reco, MODEL, RMSE_KNN

# Only now for check everything is OK
KNN_articles = pickle.load(open('KNN_articles.pkl', 'rb'))  
print(KNN_articles)


def get_recommandation(predictions, n=5):
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
    top_recommandation = defaultdict(list)
    for uid, iid, r_ui, est, _ in predictions:
        top_recommandation[uid].append((iid, est))

    # Trier les prédictions pour chaque utilisateur 
    # Récupérer les k prédictions les plus élevées
    for uid, user_ratings in top_recommandation.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_recommandation[uid] = user_ratings[:n]

    return top_recommandation


top_recommandation = get_recommandation(predictions, n=5)

# Save top_recommandation as a pickle object    
with open("top_recommandation.txt", "wb") as fp:
    pickle.dump(top_recommandation, fp)


def find_recommandation(dico, user_id):
    results = []
    query = dico[user_id]
    for uid, user_ratings in query:
        results.append(uid)
    return results


#Tester sur le user 458
print(find_recommandation(top_recommandation, 458))



