# Databricks notebook source
# MAGIC %md
# MAGIC ### COVID-19 hospitalization data example
# MAGIC #### no need to change any widgets values

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from steps.config import *
from steps.ingest import *
from steps.transforms import *
from steps.evaluation import *

# COMMAND ----------

# DBTITLE 1,Set up basic parameters

# param1
# my_model_name = "feifei_model" # my_model_name grabbed from config
dbutils.widgets.text("model_name_prefix", "") # by default, it should be blank
model_name = dbutils.widgets.get("model_name_prefix") + my_model_name

# param2
dbutils.widgets.text("experiment_destination_prefix", "/Shared/mlflow_experiments/" ) # by default, it will store the mlflow experiments in this shared location; note, create this folder, also a staging folder that is only managed by the SP
experiment_destination = dbutils.widgets.get("experiment_destination_prefix") + model_name

# param3
# model_version_to_submit = "2" # the final version of the local model that the user would like to submit to PR

# param4
# train_data_set_name = "final_train"+"_"+model_name+"_"+model_version_to_submit # the exact data set used for training to generate the final model that the DS would like to PR for

# print("param1 model_name: ", model_name)
# print("param2 experiment_destination: ", experiment_destination)
# print("param3 model_version_to_submit: ", model_version_to_submit)
# print("param4 train_data_set_name: ", train_data_set_name)

# COMMAND ----------

# DBTITLE 1,Create train and save train as delta table

# transformation
df = load_dataset() 
df = filter_country(df, country='USA') 
df = pivot_and_clean(df, fillna=0)  
df = clean_spark_cols(df)
df = index_to_col(df, colname='date')
df = drop_na (df)


# convert from Pandas to a pyspark sql DataFrame; generate a new feature
from pyspark.sql import functions as F
df_train = spark.createDataFrame(df)
df_train = df_train.select(
    '*',
    F.quarter('date').alias('quarter')
)

# featurization and select features
feature_cols = ['Daily_ICU_occupancy', 'Daily_hospital_occupancy', 'quarter']
y_col = 'Weekly_new_hospital_admissions' # must specify which y column it is
df_train = df_train.select(* (feature_cols+[y_col])).withColumnRenamed(y_col, "y")
display(df_train)



# COMMAND ----------

# DBTITLE 1,Train a model, MLflow exp, register model
import xgboost
from xgboost import XGBRegressor
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle

pd_train = df_train.toPandas().sort_values(by=['Daily_ICU_occupancy', 'Daily_hospital_occupancy'])

X_train = pd_train[feature_cols]
y_train = pd_train[['y']]
experiment = mlflow.set_experiment(experiment_destination) # set experiment

with mlflow.start_run(run_name='my_run_xgboost') as run:
  num_estimators = 1000
  np.random.seed(0)
  model = XGBRegressor(n_estimators=num_estimators, max_depth=7, eta=0.1, random_state=0, seed = 1000)  #, subsample=0.7, colsample_bytree=0.8, colsample_bylevel=0.8, colsample_bynode=0.8
  model.fit(X_train, y_train)

  mlflow.log_param('n_estimators', num_estimators)
  y_pred = model.predict(X_train)
  accuracy = evaluate_accuracy(model, X_train, y_train)
  mlflow.log_metric('accuracy', accuracy)
  
  signature = infer_signature(X_train, y_pred)
  # MLflow contains utilities to create a conda environment used to serve models.
  # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(xgboost.__version__)],
        additional_conda_channels=None,
    )

  run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "my_run_xgboost"').iloc[0].run_id
  mlflow.xgboost.log_model(registered_model_name=model_name, xgb_model=model, conda_env=conda_env, signature=signature, input_example=X_train, artifact_path=f"runs:/{run_id}/{model_name}") 


# COMMAND ----------

# DBTITLE 1,Write train to Delta Lake

np.random.seed(0)
model_version_for_train = func_grab_newest_version_number_of_a_model (model_name, stage="None")
train_data_set_name_to_save = "final_train"+"_"+model_name+"_"+model_version_for_train # the exact data set used for training to generate the final model that the DS would like to PR for
df_train.write.mode('overwrite').option('overwriteSchema', True).saveAsTable(train_data_set_name_to_save) # delta table, must have a column called "y"
print("train_data_set_name_to_save: ", train_data_set_name_to_save)
display(df_train)

# COMMAND ----------

# DBTITLE 1,Optional cell: check the prediction result
pd_results = X_train.copy()
pd_results['y_hats'] = y_pred 
pd_results['y'] = y_train 
display(pd_results)

# COMMAND ----------

print("finished training !!!")

# COMMAND ----------

print("latest 'None' model version of ", model_name, "is", model_version_for_train)
print("param1 model_name: ", model_name)
print("param2 experiment_destination: ", experiment_destination)
print("param3 model_version_to_submit: ", model_version_to_submit)
print("param4 train_data_set_name: ", train_data_set_name)
print("param5 accuracy: ", accuracy) # to delete: 618.197331327945,  600.8278826822209, 525.0411580787095

# COMMAND ----------

#dbutils.widgets.removeAll()

# COMMAND ----------


