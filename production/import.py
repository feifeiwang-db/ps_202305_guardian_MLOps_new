# Databricks notebook source
# MAGIC %pip install mlflow-export-import

# COMMAND ----------

# DBTITLE 1,Import and set up
import steps.config
from steps.config import *

import mlflow
import pyspark
from pyspark.sql.functions import current_timestamp

client = mlflow.client.MlflowClient()

# COMMAND ----------

# DBTITLE 1,All functions
def get_notebook_formats():
    notebook_formats = 'SOURCE'
    notebook_formats = notebook_formats.split(",")
    if "" in notebook_formats: notebook_formats.remove("")
    return notebook_formats

def func_todaysdate():
  '''
  return todays date in string format, such as "20230329" in PST time zone
  '''
  from datetime import datetime, timedelta
  now_EST = datetime.today() - timedelta(hours=8)
  todaysdate = now_EST.strftime('%Y_%m_%d')
  todaysdate = todaysdate.replace("_", "")
  return todaysdate

def func_summary_table_if_exists_already(external_location: str, table_name: str):
  '''
  param: external_location: the folder/container name that stores the summary delta table
  param: table_name: the table name that stores the summary delta table
  rtn: bool_summary_table_exists_already boolean: check if this summary table already exists; If not, create an empty table with schema; if yes, return True
  rtn: summary_table_location string: the external location for where this summary delta table is 
  '''
  # from delta.tables import *
  # from pyspark.sql.functions import * 
  from delta.tables import DeltaTable
  summary_table_location = external_location+"/"+table_name
  try:
      dbutils.fs.ls(external_location)
  except Exception as ex:
      if "FileNotFoundException" in str(ex):
          dbutils.fs.mkdirs(external_location) 
          print("!!!! just created the external_location", external_location)
  bool_summary_table_exists_already = DeltaTable.isDeltaTable(spark, summary_table_location)
  if bool_summary_table_exists_already == False:
      # Create or replace table with path and add properties
      DeltaTable.createOrReplace(spark) \
      .addColumn("todaysdate", "STRING") \
      .addColumn("from_workspace", "STRING") \
      .addColumn("imported_model_name", "STRING") \
      .addColumn("imported_model_version", "STRING") \
      .addColumn("external_location", "STRING") \
      .addColumn("timestamp", "TIMESTAMP") \
      .location(summary_table_location) \
      .execute()
  return bool_summary_table_exists_already, summary_table_location


def func_should_export_this_production_version_or_not (model_name: str): # same function from export notebook
  '''
  This function is to determine if this model has a production version that needs to be exported.
  1. Make sure this model at least have some versions in production stage
  2. If "1" is yes, then check if the latest production version's description contains "Exported Already On". If contains, then should not export this model again, if no, then we should export it.
  param: model_name: string, the model's name
  rtn: bool_should_export_this_production_version: boolean. Yes means
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
      print("Though this model:", model_name,"has latest Production stage in version",latest_production_version, ", it has previously already been exported. Thus, no actions here.")  
      print("\n")
  return bool_should_export_this_production_version, latest_production_version, latest_description_production_version



def func_list_models_in_production(bool_print_info=False): # same function from export notebook
  '''
  This function is to find all existing model_name in the current workspace's model registry that at least has a Production stage.
  var: bool_print_info: boolean variable, True to print details; False not to print details. 
  rtn: list_models_having_production:list of model names that have "production"
  '''
  from pprint import pprint
  list_models_having_production = []
  #list_models_expect_to_export = []
  for rm in client.search_registered_models():
      dict_current_model = dict(rm)
      list_model_stages = []
      current_model_name = dict_current_model['name']
      num_latest_versions = len(dict_current_model['latest_versions']) # the count of diverse stages that this model owns --- a subset of [None, Archive, Staging, Production]
      if bool_print_info:
        print("current_model_name: ", current_model_name)
        print("num distinct latest versions: ", num_latest_versions)
      for i in range(0, num_latest_versions):
        current_stage = dict_current_model['latest_versions'][i].current_stage
        current_version = dict_current_model['latest_versions'][i].version
        current_description = dict_current_model['latest_versions'][i].description
        list_model_stages = list_model_stages + [current_stage]
        if bool_print_info:
          print("latest stage: ", current_stage,", with version: ", current_version, ", description: ", current_description)
      if "Production" in list_model_stages:
        list_models_having_production = list_models_having_production + [current_model_name]
      if bool_print_info:
        print("\n") 
  return list_models_having_production

# COMMAND ----------

# DBTITLE 1,Define hard coded variables
# variables that don't need to change often, some are hard coded here
notebook_formats = get_notebook_formats() # format is always 'SOURCE' here
todaysdate = func_todaysdate()
delete_model = False
tmp_dbfs_di_base = "dbfs:/home/mlflow_export_import_test" # internal DBFS temp path for storing files

# import summary table related vars
external_import_summary_location = "%s/import_summary_delta"% (external_location_folder)
external_import_summary_table_name = "import_summary.delta"
temp_import_internal_table_name = "table_full_import_summary"

external_export_summary_location = "%s/export_summary_delta"% (external_location_folder)
external_export_summary_table_name = "export_summary.delta"

# COMMAND ----------

# DBTITLE 1,Grab current export summary 
from pyspark.sql.functions import col
from pyspark.sql import Window
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType

summary_table_location = "%s/%s" %(external_export_summary_location, external_export_summary_table_name)
df_export_summary = spark.read.load(path = summary_table_location).orderBy(col('timestamp').desc())
display(df_export_summary)


print("filtered result: (if Query returned no results, that means there is nothing new to import today; else, below are the models that will be imported)")
# complex filter logic: to only the versions that need to be imported
# generate a filtered dataframe for only models from today with latest model version per workspace per model, if this df is not empty, then do the rest of import steps. 
# df_export_summary2 = df_export_summary.withColumn("exported_model_version", df_export_summary["exported_model_version"].cast(IntegerType()))
# df_to_import = df_export_summary2.filter(col('todaysdate')== todaysdate)
# w = Window.partitionBy('from_workspace', 'exported_model_name')
# df_to_import_filtered = df_to_import.withColumn('maxB', f.max('timestamp').over(w)).where(f.col('timestamp') == f.col('maxB')).drop('maxB')
# bool_if_nothing_to_import = (len(df_to_import_filtered.head(1)) == 0)
# display(df_to_import_filtered)

# for simplicity, here we will always import the ONLY ONE model and version based on the latest timestamp from the export summary delta table
df_to_import_filtered = df_export_summary.limit(1)
bool_if_nothing_to_import = False
display(df_to_import_filtered)

# COMMAND ----------

# DBTITLE 1,Import

bool_summary_table_exists_already, import_summary_table_location = func_summary_table_if_exists_already (external_import_summary_location, external_import_summary_table_name)
list_current_prod_models = func_list_models_in_production()
#print("list_current_prod_models before import action: ", list_current_prod_models)
print("if there is nothing to import: ", bool_if_nothing_to_import)
if bool_if_nothing_to_import == False: # if there are something need to be imported
  # store the export information into summary table
  # 1. write latest summmary info into this internal table
  if bool_summary_table_exists_already == True:
    df_summary = spark.read.load(path = import_summary_table_location)
    df_summary.write.mode("overwrite").saveAsTable(temp_import_internal_table_name)

  # loop over each row to get model name and version that needs to be exported
  pd_df_to_import_filtered = df_to_import_filtered.toPandas()
  for index, row in pd_df_to_import_filtered.iterrows():
    original_workspace_id = row['from_workspace'].split("_")[1]
    original_model_name = row['exported_model_name']
    external_input_dir_original = row['external_location']
    external_input_dir = external_input_dir_original.rsplit("/", 1)[0] # do not include the last part of the run_id, the import tool import all experiments under the same model
    latest_production_version = row['exported_model_version']
    model_name = "imported_" +  original_model_name #"imported_" + original_workspace_id + "_" +  original_model_name
    tmp_dbfs_dir = "%s/workspace_%s/%s/%s" %(tmp_dbfs_di_base, original_workspace_id, todaysdate, original_model_name)
    experiment_name = "/Shared/mlflow_experiments/exp_%s" %(model_name)

    # before importing this model's new production version, if previously a production version already exists in production workspace, transition the former latest production version into archive stage:
    if model_name in list_current_prod_models:
        bool_should_export_this_production_version, recent_latest_production_version, latest_description_production_version = func_should_export_this_production_version_or_not(model_name)
        client.transition_model_version_stage(
        name=model_name,
        version=recent_latest_production_version,
        stage="Archived"
        )
        print("Just transitioned this model:", model_name, "production version",  recent_latest_production_version, "to 'Archived'")

    print("!!! working on importing model: ", model_name, "from ", external_input_dir)
    run_id = external_input_dir_original.split("/")[-1]
    
    # copy over from external ADLS to internal DBFS temp
    dbutils.fs.cp(external_input_dir, tmp_dbfs_dir, True)
    old_name = tmp_dbfs_dir + "/" + run_id + "/artifacts/runs" #the raw output in S3 does not contain ":"
    new_name = tmp_dbfs_dir + "/" + run_id + "/artifacts/runs:" #need to add the ":" in order to use the import tool to MLflow correctly
    dbutils.fs.mv(old_name, new_name, True) #rename that file path, so can copy fine
    import os
    os.environ["INPUT_DIR"] = tmp_dbfs_dir.replace("dbfs:","/dbfs")
    from mlflow_export_import.model.import_model import ModelImporter
    importer = ModelImporter(mlflow.client.MlflowClient())
    importer.import_model(model_name, 
                          input_dir = tmp_dbfs_dir, 
                          experiment_name = experiment_name, 
                          delete_model = delete_model)

    
    # adding the logic of constructing the import_summary table and save it as a delta to ADLS

    # 2. construct the summary information from today about newly exported models
    df_import_summary_today = spark.createDataFrame(
    [
        (todaysdate,"workspace_"+original_workspace_id, model_name, str(latest_production_version), external_input_dir_original),  # create your data here, be consistent in the types.
    ],
    ['todaysdate','from_workspace','imported_model_name','imported_model_version', 'external_location'] 
    )
    df_import_summary_today = df_import_summary_today.withColumn("timestamp", current_timestamp())
    df_import_summary_today.write.mode("append").option("mergeSchema", "true").saveAsTable(temp_import_internal_table_name)

     # read in the updated "appended" table with temp_internal_table_name; overwrite to external import delta table. Whenever insert a row, write to external delta table as well. 
    df_updated = spark.read.table(temp_import_internal_table_name)
    df_updated.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(import_summary_table_location)
    print("\n") 

# COMMAND ----------

# DBTITLE 1,Check the import summary table
from pyspark.sql.functions import col
df_summary = spark.read.load(path = import_summary_table_location)
display(df_summary.orderBy(col('timestamp').desc(), col('todaysdate').desc(), col('from_workspace'), col('imported_model_name') ))
