import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/mashabelaworkers/End_To_End_MLops_Water_Portability_Prediction_Project.mlflow")

import dagshub
dagshub.init(repo_owner='mashabelaworkers', repo_name='End_To_End_MLops_Water_Portability_Prediction_Project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)