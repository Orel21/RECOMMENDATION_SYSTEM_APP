from flask import Flask, render_template, request, jsonify, render_template_string, redirect, url_for
import pandas as pd
import numpy as np
import pickle as pkl
import json
import os
import flask_monitoringdashboard as dashboard
from IA.model import get_data, prepro, training_model, get_predictions, get_recommendation, find_recommendation


# Create application
app = Flask(__name__)
# Monitoring
dashboard.bind(app)


# ----------------------------------------------------------
# PREREQUISITES
# ----------------------------------------------------------

# deployment AI from CSV 
def executing_training_model():

    # Get data & Preprocessing
    articles_embeddings, df_clicks, df_articles_metadata = get_data()
    df_dropped = prepro(articles_embeddings, df_clicks, df_articles_metadata)

    # Get model:
    testset = training_model(df_dropped)
    MODEL = pkl.load(open('KNN_articles.pkl', 'rb'))

    # Get predictions
    predictions, df_reco, RMSE_KNN = get_predictions(testset, MODEL)
    print("training_model OK")
    
    # Get recommendations
    top_recommendation = get_recommendation(predictions, n=5)
    print("top_recommendation OK", type(top_recommendation), top_recommendation[458])

    return df_dropped, predictions, top_recommendation, MODEL, RMSE_KNN

 
# if not os.path.isfile("KNN_articles.pkl", ):
#     #Generate the pickle's file with the model trained
#     print("if file is not found, start 'executing_get_recommendation':")
#     _, predictions, top_recommendation, MODEL, _ = executing_training_model()
# else: 
#    #Load model
#    print("if File is found, load 'KNN_articles.pkl'") 
#    MODEL = pkl.load(open('KNN_articles.pkl','rb'))


# ------------------------------------------
# ROUTING
#-------------------------------------------

#Link to the HOMEPAGE (endpoint /)
@app.route('/')
def home():

    df_dropped = pd.read_csv("df_dropped.csv")

    # Create a list of the best articles' rating for inserting into a carrousel's recommendation
    df_best_session = df_dropped[df_dropped.session_size >= 5].sort_values(by="session_size", ascending=False)
    print("df_best_session", df_best_session.head())
    print("df_best_session shape", df_best_session.shape)

    best_session_embeddings = []
    for x in df_best_session.itertuples():
        best_session_embeddings.append(x.articles_embeddings)
        #Show only the first embedding for testing
        print(best_session_embeddings[0])

    # Get user_id for selector
    options_id_for_dropdown = []
    for idx in enumerate(df_dropped.user_id.unique()):
        options_id_for_dropdown.append(idx[1])
        # Order list by ascending values
        options_id_for_dropdown= sorted(options_id_for_dropdown)
    
    # get category_id for adding recommendations' details in modal
    get_category_id = []
    for cat in enumerate(df_dropped.category_id.unique()):
        get_category_id.append(cat[1])
        # Order list by ascending values
        get_category_id = sorted(get_category_id)
    
    # Create a list of the best articles' rating for inserting into a recommendations' carrousel
    df_best_session = df_dropped[df_dropped.session_size >= 5].sort_values(by="session_size", ascending=False)
    print("df_best_session", df_best_session.head())
    print("df_best_session shape", df_best_session.shape)

    best_session_embeddings = []
    for x in df_best_session.itertuples():
        best_session_embeddings.append(x.articles_embeddings)
        #Show only the first embedding for testing
        print(best_session_embeddings[0])

    # Get category_id from best_session
    get_category_best_session = []
    for cat_best in enumerate(df_best_session.category_id.unique()):
        get_category_best_session.append(cat_best[1])
        get_category_best_session =get_category_id[:10]
        # Order list by ascending values
        get_category_best_session = sorted(get_category_best_session)  

    return render_template("index.html", best_session_embeddings=best_session_embeddings, options_id_for_dropdown = options_id_for_dropdown,
     get_category_id=get_category_id, get_category_best_session=get_category_best_session)


@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    df_dropped = pd.read_csv("df_dropped.csv")

    # get user_id for selector
    options_id_for_dropdown = []
    for idx in enumerate(df_dropped.user_id.unique()):
        options_id_for_dropdown.append(idx[1])
        # Order list by ascending values
        options_id_for_dropdown = sorted(options_id_for_dropdown)
    
    # get article_id for adding recommendations' details in modal
    get_article_id = []
    for art in enumerate(df_dropped.article_id.unique()):
        get_article_id.append(art[1])
        # Order list by ascending values
        get_article_id = sorted(get_article_id)

    # get publisher_id for adding recommendations' details in modal
    get_publisher_id = []
    for pub in enumerate(df_dropped.publisher_id.unique()):
        get_publisher_id.append(pub[1])
        # Order list by ascending values
        get_publisher_id = sorted(get_publisher_id)
  
    # get category_id for adding recommendations' details in modal
    get_category_id = []
    for cat in enumerate(df_dropped.category_id.unique()):
        get_category_id.append(cat[1])
        # Order list by ascending values
        get_category_id = sorted(get_category_id)

    # get words_count for adding recommendations' details in modal
    get_words_count = []
    for word in enumerate(df_dropped.category_id.unique()):
        get_words_count.append(word[1])
    
    _, _, top_recommendation, _, _ = executing_training_model()

    if request.method == 'POST':
        ENDPOINT = "https://booksapp-reco.azurewebsites.net"
        input_user = request.form.get('input_user') 
        input_user = int(input_user)
        print("input_user", input_user)
        reco_for_user = request.form.get(ENDPOINT + "/predict", json = {"input_user":input_user})
        # reco_for_user = find_recommendation(top_recommendation, input_user)
        print("reco_for_user", reco_for_user)

    # Create a list of the best articles' rating for inserting into a recommendations' carrousel
    df_best_session = df_dropped[df_dropped.session_size >= 5].sort_values(by="session_size", ascending=False)
    print("df_best_session", df_best_session.head())
    print("df_best_session shape", df_best_session.shape)

    best_session_embeddings = []
    for x in df_best_session.itertuples():
        best_session_embeddings.append(x.articles_embeddings)
        #Show only the first embedding for testing
        print(best_session_embeddings[0])

    # Get category_id from best_session
    get_category_best_session = []
    for cat_best in enumerate(df_best_session.category_id.unique()):
        get_category_best_session.append(cat_best[1])
        get_category_best_session =get_category_id[:10]
        # Order list by ascending values
        get_category_best_session = sorted(get_category_best_session)   
        
    return render_template('predict.html', input_user = input_user, reco_for_user = reco_for_user, df_dropped= df_dropped,
    options_id_for_dropdown=options_id_for_dropdown, best_session_embeddings=best_session_embeddings,
    get_article_id=get_article_id, get_category_id=get_category_id, get_publisher_id=get_publisher_id, get_words_count=get_words_count,
    get_category_best_session=get_category_best_session)



@app.route('/library', methods = ['GET', 'POST'])
def input_bdd():

    # if request.method == 'POST':
    #     title = request.form.get('title')
    #     publisher = request.form.get('publisher')
    #     category = request.form.get('category')
    #     year = request.form.get('year_of_publication')
    #     link = request.form.get('link')
        
    #     return redirect(url_for('index'))

    return render_template("library.html")


#(debug = True) Refresh the page without always having to restart the exe
if __name__ == "__main__":
    app.run(debug = True) # use_reloader=False