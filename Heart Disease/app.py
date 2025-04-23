import pandas as pd
import numpy as np

# to visualize the dataset
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# To preprocess the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# import iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# for classification tasks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    r2_score,
    mean_squared_error,
)

# impot pipeline
from sklearn.pipeline import Pipeline

# ignore warnings
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
import joblib
from flask import Flask, render_template, request

model = joblib.load(
    r"C:\Users\navya\Desktop\MajorFinal(Heart Disease Prediction)\MajorFinal(Heart Disease Prediction)\major_finalModel5.pkl"
)


app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/normal.html")
def nextPage():
    return render_template("normal.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        Fn = request.form["fname"]
        age = request.form["ag"]
        mof = "Male"
        getgen = request.form["ge"]
        if getgen[0] == "f" or getgen[0] == "F":
            mof = "Female"
        ctype = "non-anginal"
        chestpain = request.form["cp"]
        if chestpain[1] == "s" or chestpain[1] == "S":
            ctype = "asymptomatic"
        elif chestpain[0] == "a" or chestpain[0] == "A":
            ctype = "atypical angina"
        elif chestpain[0] == "t" or chestpain[0] == "T":
            ctype = "typical angina"

        trb = request.form["trest"]
        ch = request.form["chol"]
        bfb = 1
        fbs = request.form["fb"]
        if fbs[0] == "F" or fbs[0] == "f":
            bfb = 0
        re = 0
        res = request.form["rest"]
        if res[0] == "l" or res[0] == "L":
            re = 1
        elif res[0] == "s" or res[0] == "S":
            re = 2
        thalch = request.form["t"]
        exb = 1
        exhan = request.form["ex"]
        if exhan[0] == "F" or exhan[0] == "f":
            exb = 0
        dep = request.form["de"]
        slo = request.form["sl"]
        sa = 0
        if slo[0] == "u" or slo[0] == "U":
            sa = 1
        elif slo[0] == "d" or slo[0] == "D":
            sa = 2
        c = request.form["ca"]

        b = request.form["bm"]

        inputs = [
            [
                float(age),
                mof,
                ctype,
                float(trb),
                float(ch),
                bfb,
                float(re),
                float(thalch),
                exb,
                float(dep),
                float(sa),
                float(c),
                float(b),
            ]
        ]
        print(inputs)
        results = model.predict(inputs)
        print(results)
        if str(results[0]) == "0":
            return render_template("free.html", name=Fn)
        else:
            return render_template("results.html", name=Fn)
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
