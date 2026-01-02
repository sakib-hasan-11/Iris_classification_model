import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data_processing" / "processed_iris.csv"

df = pd.read_csv(DATA_PATH)


x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


from xgboost import XGBClassifier

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,

    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,

    subsample=1.0,
    colsample_bytree=1.0,

    min_child_weight=1,
    gamma=0,

    reg_alpha=0,
    reg_lambda=1,

    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train,y_train)


print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model,'model.pkl')


# about the model 


# Train accuracy: 1.0
# Test accuracy: 0.9
# Accuracy: 0.9
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        10
#            1       0.77      1.00      0.87        10
#            2       1.00      0.70      0.82        10

#     accuracy                           0.90        30
#    macro avg       0.92      0.90      0.90        30
# weighted avg       0.92      0.90      0.90        30


