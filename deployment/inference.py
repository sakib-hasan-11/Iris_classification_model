import json
import joblib
import pandas as pd
from pathlib import Path


# -------------------------------------------------
# 1. Load model (called once when container starts)
# -------------------------------------------------
def model_fn(model_dir):
    """
    Load the trained model from the model directory
    """
    model_path = Path(model_dir) / "iris_xgboost_model.pkl"
    model = joblib.load(model_path)
    return model


# -------------------------------------------------
# 2. Parse incoming request
# -------------------------------------------------
def input_fn(request_body, request_content_type):
    """
    Deserialize input data
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)

        # Expecting input as list of feature values
        # Example:
        # {
        #   "features": [5.1, 3.5, 1.4, 0.2]
        # }

        df = pd.DataFrame(
            [data["features"]],
            columns=[
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm"
            ]
        )
        return df

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


# -------------------------------------------------
# 3. Run prediction
# -------------------------------------------------
def predict_fn(input_data, model):
    """
    Make prediction using the model
    """
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    return {
        "prediction": int(prediction[0]),
        "probabilities": probabilities[0].tolist()
    }


# -------------------------------------------------
# 4. Format output response
# -------------------------------------------------
def output_fn(prediction, response_content_type):
    """
    Serialize prediction output
    """
    if response_content_type == "application/json":
        label_map = {
            0: "setosa",
            1: "versicolor",
            2: "virginica"
        }

        response = {
            "predicted_class": label_map[prediction["prediction"]],
            "class_index": prediction["prediction"],
            "probabilities": prediction["probabilities"]
        }

        return json.dumps(response), response_content_type

    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
