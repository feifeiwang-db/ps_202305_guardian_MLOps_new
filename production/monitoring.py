# Databricks notebook source
import steps.config
from steps.config import *
from steps.evaluation import *

# COMMAND ----------

def func_should_export_this_production_version_or_not (model_name: str): ###common from export
  '''
  This function is to determine if this model has a production version that needs to be exported.
  1. Make sure this model at least have some versions in production stage
  2. If "1" is yes, then check if the latest production version's description contains "Exported Already On". If contains, then should not export this model again, if no, then we should export it.
  param: model_name: string, the model's name
  rtn: bool_should_export_this_production_version: boolean. Yes means should export; latest_production_version: str, the latest version number of production version of the model 
  '''
  bool_should_export_this_production_version = True
  list_production_versions = []
  dict_production_verions = dict()
  for mv in client.search_model_versions(f"name='{model_name}'"): #check all current versions with "Production" stage
      if mv.current_stage == "Production":  
          vversion = int(mv.version)
          list_production_versions = list_production_versions + [vversion]
          current_description = mv.description
          dict_production_verions[vversion] = current_description
          # print("Production version: ", vversion)
          # print("current description: ", current_description) 
  if list_production_versions==[]:
    bool_should_export_this_production_version = False
    latest_production_version = "-1000"
    latest_description_production_version = "error, this model does not have any production stages and should not be exported"
    print("This model:", model_name, "does not have any Production stage versions") # this is a guardrill double check, because we have previously already filtered out models that do not have production stages
  else:
    latest_production_version_int = max(list_production_versions)
    latest_description_production_version = dict_production_verions[latest_production_version_int]
    latest_production_version = str(latest_production_version_int)
    #print("latest_production_version: ", latest_production_version)
    #print("latest_description_production_version: ", latest_description_production_version)
    if "Exported Already On" in latest_description_production_version:
      bool_should_export_this_production_version = False  
      print("Though this model:", model_name,"has latest Production stage in version",latest_production_version, ", it has previously already been exported. Thus, no actions here. If you still want to export this version again, please delete or modify the description of this version, and rerun code/pipeline")  
      print("\n")
  return bool_should_export_this_production_version, latest_production_version, latest_description_production_version

# COMMAND ----------

import pandas as pd

dbutils.widgets.text("model_name_prefix", "") # by default, it should be blank
model_name = "imported_"+ dbutils.widgets.get("model_name_prefix") + my_model_name

current_artifact_uri = f"models:/{model_name}/Production"

bool_should_export_this_production_version, latest_production_version, latest_description_production_version = func_should_export_this_production_version_or_not (model_name) 
previous_model_version = str(int(latest_production_version) - 1)

if int(latest_production_version) >= 2: 
  previous_artifact_uri =  f"models:/{model_name}/{previous_model_version}"
  print("current model version: ", latest_production_version)
  print("previous model version: ", previous_model_version)
else:
  raise Exception("There are currently less than 2 production verions of the model, cannot compare them")


test_json = mlflow.artifacts.load_dict(current_artifact_uri + "/input_example.json") 
pd_input_example = pd.DataFrame(test_json['data'], columns=test_json['columns']) 
pd_train = pd_input_example.sort_values(by=['Daily_ICU_occupancy', 'Daily_hospital_occupancy', 'quarter'])
X_train = pd_train[['Daily_ICU_occupancy', 'Daily_hospital_occupancy', 'quarter']]
y_train = pd_train[['y']]

model1 =  mlflow.xgboost.load_model(current_artifact_uri)
model2 =  mlflow.xgboost.load_model(previous_artifact_uri)

accuracy_model1 = evaluate_accuracy(model1, X_train, y_train)
accuracy_model2 = evaluate_accuracy(model2, X_train, y_train)

# COMMAND ----------

val_diff_accuracy_betwee_2_models = abs((accuracy_model1 - accuracy_model2)/accuracy_model1)
if val_diff_accuracy_betwee_2_models<0.01:
  print("The difference is", val_diff_accuracy_betwee_2_models, ", accuracy for both models are about",accuracy_model1)

elif accuracy_model1>accuracy_model2:
  print ("Warning!!!!! DS pushed a version of production model that is having worse accuracy than the previous version")

else:
  print ("DS pushed version of production model that is having better accuracy than the previous version")

print('accuracy_model1:', accuracy_model1, ", for current model: ", current_artifact_uri)
print('accuracy_model2:', accuracy_model2, ", for previous model: ", previous_artifact_uri)
