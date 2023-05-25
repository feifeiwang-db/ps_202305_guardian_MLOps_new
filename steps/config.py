import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
client = MlflowClient()

data_path = '../raw_data/covid_hospital.csv' # original source came from 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/hospitalizations/covid-hospitalizations.csv', but it is updated twice a day, not static; Thus, for testing purpose, using a static csv file here
print(f'Data path: {data_path}')

# param1
my_model_name = "feifei_covid_model"
print("my_model_name: ", my_model_name)

# param3
model_version_to_submit = "6" # the final version of the local model that the user would like to submit to PR
print("model_version_to_submit: ", model_version_to_submit)

# param4
train_data_set_name = "final_train"+"_"+my_model_name+"_"+model_version_to_submit # the exact data set used for training to generate the final model that the DS would like to PR for
print("train_data_set_name: ", train_data_set_name)

external_location_folder = "s3a://one-env/feifei_wang/mlfow_export_import_new" 
print("external_location_folder: ", external_location_folder)

def func_grab_newest_version_number_of_a_model (model_name, stage="None"): 
  '''
  Get the latest version number (string) of a specificed model and its stage
  :param: model_name: string: the name of the model
  :param: stage: stage of the model, "None", "Archive", "Staging", "Production"
  :return val_str_version: string of the latest version of the model in the specificed stage, for example: '9' in "None" stage. 
  '''
  val_int_version = 0
  for mv in client.search_model_versions(f"name='{model_name}'"):
    dict_current_model = dict(mv)
    current_stage = dict_current_model['current_stage']
    current_version = dict_current_model['version']
    if current_stage == stage:
      val_int_version = max(val_int_version, int(current_version))
  if val_int_version == 0:
    raise Exception ("ERROR!!! no versions of your model: ", model_name, "are in the stage: ", stage)
  else:
    val_str_version = str(val_int_version)
    return val_str_version

def func_grab_model_based_on_name_and_version (model_name, model_version):
  '''
  :param: model_name: string: the name of the model
  :param: model_version: string: the version of the model
  :return model: the MLflow model
  '''
  model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
  return model


# def func_grab_workspace_name():
#   '''
#   return the workspace name
#   '''
# #   list_info_workspace = eval(spark.conf.get("spark.databricks.clusterUsageTags.clusterAllTags"))
# #   for i in range(0, len(list_info_workspace)):
# #     dict_i = list_info_workspace[i]
# #     if dict_i['key'] == 'dd_workspace':
# #       return dict_i['value']  #in Azure
# #   return "unknown"

#   return spark.conf.get("spark.databricks.workspaceUrl").split('.')[0] #in AWS
