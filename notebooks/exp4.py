import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.models import infer_signature
import dagshub


# Initialize DagsHub MLflow tracking integration
dagshub.init(
    repo_owner='mashabelaworkers',
    repo_name='End_To_End_MLops_Water_Portability_Prediction_Project',
    mlflow=True
)

# Set MLflow tracking URI before setting experiment
mlflow.set_tracking_uri(
    "https://dagshub.com/mashabelaworkers/End_To_End_MLops_Water_Portability_Prediction_Project.mlflow"
)

# Use a new unique experiment name that is not deleted, to avoid errors
mlflow.set_experiment(experiment_name="Experiment_new4_unique")

# Load dataset and split into train and test sets
data = pd.read_csv(
    "/workspaces/End_To_End_MLops_Water_Portability_Prediction_Project/End_To_End_MLops_Water_Portability_Prediction_Project/data/raw/water_potability.csv"
)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Fill missing values with mean without chained assignment warning
def fill_missing_with_mean(df):
    for col in df.columns:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df.loc[:, col] = df[col].fillna(mean_val)
    return df

train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

X_train = train_processed_data.drop(columns=["Potability"])
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns=["Potability"])
y_test = test_processed_data["Potability"]

# Define Random Forest model and hyperparameter distribution
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [None, 4, 5, 6, 10],
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

with mlflow.start_run(run_name="Random_Forest_Tuning") as parent_run:
    random_search.fit(X_train, y_train)

    # Log hyperparameter combinations as nested runs
    for i, params in enumerate(random_search.cv_results_['params']):
        with mlflow.start_run(run_name=f"Combination_{i+1}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])

    print("Best parameters found:", random_search.best_params_)

    mlflow.log_params(random_search.best_params_)

    best_rf = random_search.best_estimator_
    best_rf.fit(X_train, y_train)

    # Save the best model to a pickle file locally
    pickle.dump(best_rf, open("model.pkl", "wb"))

    # Load the model and predict on test data
    model = pickle.load(open("model.pkl", "rb"))
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log performance metrics to MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log train and test datasets using MLflow data API (2.7+)
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)
    mlflow.log_input(train_df, "train")
    mlflow.log_input(test_df, "test")

    # Optional: log this source file (uncomment when running as a script)
    # mlflow.log_artifact(__file__)

    # Infer and log model signature to improve reproducibility
    signature = infer_signature(X_test, model.predict(X_test))

    # Due to DagsHub current API limitation, log model pickle artifact instead of calling mlflow.sklearn.log_model
    mlflow.log_artifact("model.pkl")

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
