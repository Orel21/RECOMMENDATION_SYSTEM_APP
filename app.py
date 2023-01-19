from flask import Flask, render_template, request, jsonify, render_template_string, redirect
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

#_, predictions, top_recommendation, MODEL, RMSE_KNN = executing_training_model()

#Testing on user 458
# user = 458
# reco_for_user = find_recommendation(top_recommendation, user)
# print("Reco for user", user, ":", reco_for_user)

# print("testing app")


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

    return render_template("index.html", best_session_embeddings=best_session_embeddings, options_id_for_dropdown = options_id_for_dropdown) 


@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    df_dropped = pd.read_csv("df_dropped.csv")

    # Get user_id for selector
    options_id_for_dropdown = []
    for idx in enumerate(df_dropped.user_id.unique()):
        options_id_for_dropdown.append(idx[1])
        # Order list by ascending values
        options_id_for_dropdown= sorted(options_id_for_dropdown)
    
    _, predictions, top_recommendation, MODEL, RMSE_KNN = executing_training_model()

    if request.method == 'POST':
        input_user = 458
        #input_user = request.args.get("input_user") 
        print("input_user", input_user)

        reco_for_user = find_recommendation(top_recommendation, input_user)
        
    return render_template('index.html', input_user = input_user, reco_for_user = reco_for_user, df_dropped= df_dropped, options_id_for_dropdown=options_id_for_dropdown)
    # return redirect('/')  


#(debug = True) Refresh the page without always having to restart the exe
if __name__ == "__main__":
    app.run(debug = True) # use_reloader=False