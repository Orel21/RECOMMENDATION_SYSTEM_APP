from flask import Flask, render_template, request, render_template_string
import pandas as pd
import numpy as np
import pickle
import json
import os


# Create application
app = Flask(__name__)

# ------------------------------------------
# ROUTING
#-------------------------------------------

#Link to the HOMEPAGE (endpoint /)
@app.route('/')
def home():
    return render_template("index.html")


#(debug = True) Refresh the page without always having to restart the exe
if __name__ == "__main__":
    app.run(debug = True)
