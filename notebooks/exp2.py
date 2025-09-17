# exp2.py - Fixed MLflow experiment tracking with DagsHub and multiple models

import pandas as pd
import mlflow
import dagshub
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Initialize DagsHub integration and set MLflow tracking URI first
dagshub.init(
    repo_owner='mashabelaworkers',
    repo_name='End_To_End_MLops_Water_Portability_Prediction_Project',
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/mashabelaworkers/End_To_End_MLops_Water_Portability_Prediction_Project.mlflow"
)

# Use a new experiment name or one that exists/not deleted
mlflow.set_experiment(experiment_name="Experiment_new2")

# Load dataset and split into train and test
data = pd.read_csv(
    "/workspaces/End_To_End_MLops_Water_Portability_Prediction_Project/End_To_End_MLops_Water_Portability_Prediction_Project/data/raw/water_potability.csv"
)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df.loc[:, column] = df[column].fillna(median_value)
    return df


train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

X_train = train_processed_data.drop(columns=["Potability"])
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns=["Potability"])
y_test = test_processed_data["Potability"]

# Define models to evaluate
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Random_Forest": RandomForestClassifier(),
    "Support_Vector_Classifier": SVC(probability=True),
    "Decision_Tree": DecisionTreeClassifier(),
    "K_Nearest_Neighbors": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

with mlflow.start_run(run_name="Water_Potability_Models_Experiment"):
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            model.fit(X_train, y_train)
            
            # Save model with pickle
            model_filename = f"{model_name}.pkl"
            pickle.dump(model, open(model_filename, "wb"))
            
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Confusion matrix plot
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            plt.savefig(f"confusion_matrix_{model_name}.png")
            plt.close()
            
            mlflow.log_artifact(f"confusion_matrix_{model_name}.png")
            
            # Instead of mlflow.sklearn.log_model (not supported on DagsHub), log pickle artifact
            mlflow.log_artifact(model_filename)
            
            # Log source code for reproducibility if running as script
            # mlflow.log_artifact(__file__)
            
            mlflow.set_tag("author", "Workers Mashabela")

print("All models trained and logged successfully.")
