# exp1.py - Complete end-to-end MLflow tracking example with DagsHub integration

# Import necessary libraries for data handling, machine learning, tracking, and visualization
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize DagsHub and set up MLflow experiment tracking
dagshub.init(
    repo_owner='mashabelaworkers',
    repo_name='End_To_End_MLops_Water_Portability_Prediction_Project',
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/mashabelaworkers/End_To_End_MLops_Water_Portability_Prediction_Project.mlflow"
)

# Use a new experiment name or ensure "Experiment 1" is restored
mlflow.set_experiment("Experiment_New")

# Load the dataset from a CSV file
data = pd.read_csv(
    "/workspaces/End_To_End_MLops_Water_Portability_Prediction_Project/End_To_End_MLops_Water_Portability_Prediction_Project/data/raw/water_potability.csv"
)

# Split the dataset into training and test sets (80% training, 20% testing)
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

# Define a function to fill missing values in the dataset with the median value of each column
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():  # Check if there are missing values in the column
            median_value = df[column].median()  # Calculate the median
            df[column].fillna(median_value, inplace=True)  # Fill missing values with the median
    return df

# Fill missing values in both the training and test datasets using the median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Separate features (X) and target (y) for training
X_train = train_processed_data.drop(columns=["Potability"], axis=1)  # Features
y_train = train_processed_data["Potability"]  # Target variable

X_test = test_processed_data.drop(columns=["Potability"], axis=1)  # Features for testing
y_test = test_processed_data["Potability"]  # Target variable for testing

n_estimators = 100  # Number of trees in the Random Forest

# Start a new MLflow run for tracking the experiment
with mlflow.start_run():

    # Initialize and train the Random Forest model
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)

    # Save the trained model to a file using pickle
    pickle.dump(clf, open("model.pkl", "wb"))

    # Predict the target for the test data
    y_pred = clf.predict(X_test)

    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)  # Accuracy
    precision = precision_score(y_test, y_pred)  # Precision
    recall = recall_score(y_test, y_pred)  # Recall
    f1 = f1_score(y_test, y_pred)  # F1-score

    # Log metrics to MLflow for tracking
    mlflow.log_metric("acc", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score", f1)

    # Log the number of estimators used as a parameter
    mlflow.log_param("n_estimators", n_estimators)

    # Generate a confusion matrix to visualize model performance
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()  # Close the plot to avoid memory issues

    # Log the confusion matrix image to MLflow
    mlflow.log_artifact("confusion_matrix.png")

    # Log the trained model to MLflow
    #mlflow.sklearn.log_model(clf, "RandomForestClassifier")
    # Save model using pickle
    pickle.dump(clf, open("model.pkl", "wb"))

    # Log the pickle file manually instead of mlflow.sklearn.log_model
    mlflow.log_artifact("model.pkl")


    # Optionally log the source code file if running as a script
    # mlflow.log_artifact("exp1.py")

    # Set tags in MLflow to store additional metadata
    mlflow.set_tag("author", "Workers_Mashabela")
    mlflow.set_tag("model", "RandomForest")

# Print out the performance metrics for reference
print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
