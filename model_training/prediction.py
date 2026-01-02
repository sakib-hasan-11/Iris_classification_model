import joblib 
import pandas as pd


def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    model = joblib.load('model.pkl')
    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
    )

    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]

    label_map = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }

    print("Predicted species:", label_map[pred])
    print("Class probabilities:", proba)

# Test
predict_iris(5.1, 3.5, 1.4, 0.2)



predict_iris(5.1, 3.5, 1.4, 0.2)

