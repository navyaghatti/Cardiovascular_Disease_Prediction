import joblib
from flask import Flask, render_template, request
import pickle

model = joblib.load(
    r"C:\Users\navya\Desktop\MajorFinal(Heart Disease Prediction)\MajorFinal(Heart Disease Prediction)\major_finalModel4.pkl"
)

print(type(model))

results = model.predict(
    [
        [
            12.0,
            "Male",
            "atypical angina",
            200.0,
            126.0,
            1,
            2.0,
            82.0,
            1,
            5.0,
            1.0,
            3.0,
            30.0,
        ]
    ]
)
print(results)
