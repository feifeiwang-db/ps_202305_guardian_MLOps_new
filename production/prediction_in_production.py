# Databricks notebook source
import steps.config
from steps.config import *

import pandas as pd

# COMMAND ----------

def func_predict_using_input_example_with_y (current_artifact_uri, pd_input_example): #common with 
    model = mlflow.xgboost.load_model(current_artifact_uri) 
    y_pred = model.predict(pd_input_example.drop(['y'], axis=1))
    pd_res = pd_input_example.copy()
    pd_res['y_hat'] = y_pred
    print("result is: ")
    return pd_res

# COMMAND ----------


dbutils.widgets.text("model_name_prefix", "") # by default, it should be blank
model_name = dbutils.widgets.get("model_name_prefix") + my_model_name

imported_model_name = "imported_" + model_name
current_artifact_uri = f"models:/{imported_model_name}/Production"
try: 
  test_json = mlflow.artifacts.load_dict(current_artifact_uri + "/input_example.json") #reference: https://mlflow.org/docs/latest/python_api/mlflow.artifacts.html
  pd_input_example = pd.DataFrame(test_json['data'], columns=test_json['columns']) #convert this json back to pandas df; #https://stackoverflow.com/questions/53909346/export-pandas-dataframe-to-json-and-back-to-a-dataframe-with-columns-in-the-same
  pd_res = func_predict_using_input_example_with_y (current_artifact_uri, pd_input_example)
  print("predictions based on model: ", current_artifact_uri)
  display(pd_res)
except Exception as ex:
    if "does not exist" in str(ex):
      raise Exception("Error in prediction/inference!!! The model does not exists; Or the original owner of the model did not log input_example when using MLflow, cannot find input_example.json file; Please modify the code load your own data to perform prediction")
