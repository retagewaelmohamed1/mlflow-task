import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

mlflow.set_experiment("House_Price_Prediction_Comparison")

data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

experiments = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]

for exp in experiments:
    with mlflow.start_run(run_name=exp["run_name"]):
        model = xgb.XGBRegressor(
            max_depth=exp["max_depth"],
            learning_rate=exp["learning_rate"],
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        mlflow.log_param("max_depth", exp["max_depth"])
        mlflow.log_param("learning_rate", exp["learning_rate"])

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        mlflow.xgboost.log_model(model, artifact_path="model")

        print(
            f"{exp['run_name']} Finished | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}"
        )