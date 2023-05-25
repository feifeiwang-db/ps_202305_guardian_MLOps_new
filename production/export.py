# Databricks notebook source
# MAGIC %pip install mlflow-export-import

# COMMAND ----------

import steps.config
from steps.config import *

# COMMAND ----------

# DBTITLE 1,Import and set up
import mlflow
import pyspark
from pyspark.sql.functions import current_timestamp

client = mlflow.client.MlflowClient()

# COMMAND ----------

# DBTITLE 1,Functions

def get_notebook_formats():
    notebook_formats = 'SOURCE'
    notebook_formats = notebook_formats.split(",")
    if "" in notebook_formats: notebook_formats.remove("")
    return notebook_formats

def func_should_export_this_production_version_or_not (model_name: str):
  '''
  This function is to determine if this model has a production version that needs to be exported.
  1. Make sure this model at least have some versions in production stage
  2. If "1" is yes, then check if the latest production version's description contains "Already Exported ". If contains, then should not export this model again, if no, then we should export it.
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
    if "Already Exported " in latest_description_production_version:
      bool_should_export_this_production_version = False  
      print("Though this model:", model_name,"has latest Production stage in version",latest_production_version, ", it has previously already been exported. Thus, NO ACTIONS here. If you still want to export this version again, please delete or modify the description of this version, and rerun code/pipeline; Or if you meant to transition your latest version to 'Production' stage first, then rerun code/pipeline to export it")  
      print("\n")
  return bool_should_export_this_production_version, latest_production_version, latest_description_production_version

def func_export(model_name: str, latest_production_version: str, notebook_formats: str, output_dir:str, external_location: str, run_id: str):
  '''
  This function exports the model fiels to the temp DBFS output_dir, then copy the exported files to external ADLS storage location. 
  param: model_name: the current model we are working on. Should be a model that at least have some production versions
  param: latest_production_version: the latest production version, in string format, such as "7"
  param: notebook_formats: the format of notebook to be exported. For simplicity, we always export "SOURCE" format instead of "HTML" or "JUPYTER" or "DBC".
  param: output_dir: the temporary path for dbfs file location to store the exported models
  param: external_location: the path to the AWS storage
  param: run_id: run_id for mlflow experiment
  rtn: nothing to return, but will print out messages if a model and its version is exported. 
  '''
  external_with_run_id = "%s/%s" % (external_location, run_id)
  from mlflow_export_import.model.export_model import ModelExporter
  exporter = ModelExporter(
      mlflow.client.MlflowClient(),
      notebook_formats = notebook_formats, 
      #stages = ['Production'],
      versions = [latest_production_version]
      )
  exporter.export_model(model_name = model_name, output_dir = output_dir)
  print("Finished trying to export Production stage of model to dbfs", model_name, "version ", latest_production_version)

  # copy to external location
  try:
    dbutils.fs.ls(external_location)
  except Exception as ex:
    if "FileNotFoundException" in str(ex):
      dbutils.fs.mkdirs(external_location)  
  
  old_name = output_dir + "/" + run_id + "/artifacts/runs:" #"runs:" caused issues
  new_name = output_dir + "/" + run_id + "/artifacts/runs"
  dbutils.fs.mv(old_name, new_name, True) #rename that file path, so can copy later fine without seeing errors
  dbutils.fs.cp(output_dir, external_location, True)   
  print("Exported to external_location: ", external_with_run_id)    


def func_summary_table_if_exists_already(external_location: str, table_name: str):
  '''
  param: external_location: the folder/container name that stores the summary delta table
  param: table_name: the table name that stores the summary delta table
  rtn: bool_summary_table_exists_already boolean: check if this summary table already exists; If not, create an empty table with schema; if yes, return True
  rtn: summary_table_location string: the external location for where this summary delta table is 
  '''
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
      # create or replace table with path and add properties
      DeltaTable.createOrReplace(spark) \
      .addColumn("from_workspace", "STRING") \
      .addColumn("exported_model_name", "STRING") \
      .addColumn("exported_model_version", "STRING") \
      .addColumn("external_location", "STRING") \
      .addColumn("timestamp", "TIMESTAMP") \
      .location(summary_table_location) \
      .execute()
  return bool_summary_table_exists_already, summary_table_location


def func_grab_run_id_by_model_name_version(model_name: str, latest_production_version: str):
  '''
  return the mlflow run id that generated this model_name's this version: latest_production_version
  '''
  info = client.get_registered_model(model_name)
  i = 0 
  while(i<10): #here should technically <4 is enough, None, Staging, Archive, Production
    if info.latest_versions[i].version == latest_production_version:
      run_id = info.latest_versions[i].run_id
      return run_id
    else:
      i = i + 1
  if i==10:
    print("Error: cannot find the specific latest production model version in this model name: ", latest_production_version, model_name)
  return run_id

def func_grab_workspace_name():
  '''
  return the workspace name
  '''
  return spark.conf.get("spark.databricks.workspaceUrl").split('.')[0] #in AWS


def func_check_model_having_production_versions(model_name, checkstage = 'Production'):
  '''
  return a list of model version numbers that are in a certain stage under model_name (version numbers are returned in int)
  :param: model_name, str, the name of a registered model
  :checkstage: can be one of the options of: "Production","None","Archive","Staging"
  :rtn: list_production_versions. An example of this is [16, 23]
  '''
  list_production_versions=[]
  for mv in client.search_model_versions(f"name='{model_name}'"): #check all current versions with "Production" stage
        if mv.current_stage == checkstage:  
            vversion = int(mv.version)
            list_production_versions = list_production_versions + [vversion]
            current_description = mv.description
  return list_production_versions

# COMMAND ----------

# DBTITLE 1,Define hard coded variables
# variables that don't need to change often, some are hard coded here
output_dir_base = "dbfs:/home/mlflow_export"
notebook_formats = get_notebook_formats() # format is always 'SOURCE' here
stages = 'Production'
original_workspace_name = func_grab_workspace_name()   

# summary table related vars
dbutils.widgets.text("model_name_prefix", "") # by default, it should be blank
model_name = dbutils.widgets.get("model_name_prefix") + my_model_name

external_export_summary_location = "%s/export_summary_delta"% (external_location_folder)
external_export_summary_table_name = "export_summary.delta"
temp_internal_table_name = "table_export_summary_new"
bool_summary_table_exists_already, summary_table_location = func_summary_table_if_exists_already (external_export_summary_location, external_export_summary_table_name)

print("export summary_table_location: ", summary_table_location)
print("bool_summary_table_exists_already: ", bool_summary_table_exists_already)

# COMMAND ----------

# DBTITLE 1,Export
# make sure this model has at least one production version
assert len(func_check_model_having_production_versions(model_name, "Production"))>0, f"There is no Production stage in your model: {model_name}. Please manually transition the latest tests passed Staging model into 'Production' stage, and rerun the code/pipeline"
# store the export information into summary table
# 1. write latest summmary info into this internal table
if bool_summary_table_exists_already == True:
  df_summary = spark.read.load(path = summary_table_location)
  df_summary.write.mode("overwrite").saveAsTable(temp_internal_table_name)


output_dir = "%s/%s/%s" % (output_dir_base, "workspace_"+original_workspace_name, model_name)
external_location = "%s/mlflowexportmodels/%s/%s" % (external_location_folder, "workspace_"+original_workspace_name, model_name) # modify it to workspace/date/model_name level
print("external_location for model", external_location)

bool_should_export_this_production_version, latest_production_version, latest_description_production_version = func_should_export_this_production_version_or_not (model_name) 

# export the model
if bool_should_export_this_production_version == True:
  print("Working on exporting model_name:", model_name)

  print("Output temporary dbfs dir:", output_dir)
  run_id = func_grab_run_id_by_model_name_version(model_name, latest_production_version)
  external_with_run_id = "%s/%s" % (external_location, run_id)

  func_export(model_name, latest_production_version, notebook_formats, output_dir, external_location, run_id)
  # 1. after the export, make the description of the original model's latest production version as "Already Exported ......"
  client.update_model_version( 
  name=model_name,
  version=latest_production_version,
  description="Already Exported " + ", old description: " + latest_description_production_version
  )
  # 2. construct the summary information from today about newly exported models
  df_summary_today = spark.createDataFrame(
  [
      ("workspace_"+original_workspace_name, model_name, latest_production_version, external_with_run_id),  # create your data here, be consistent in the types.
  ],
  ['from_workspace','exported_model_name','exported_model_version', 'external_location'] 
  )
  df_summary_today = df_summary_today.withColumn("timestamp", current_timestamp())
  df_summary_today.write.mode("append").option("mergeSchema", "true").saveAsTable(temp_internal_table_name)

  # read in the updated "appended" table with temp_internal_table_name; overwrite to external delta table. Whenever a new export succeeded, make sure always update the export summary
  df_updated = spark.read.table(temp_internal_table_name)
  df_updated.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(summary_table_location)
  print("\n")
  print("Has successfully exported", model_name, "model's latest production version: ", latest_production_version, ", current workspace name is:", original_workspace_name)


# COMMAND ----------

# DBTITLE 1,Check the latest summary 
from pyspark.sql.functions import col
df_summary = spark.read.load(path = summary_table_location)
display(df_summary.orderBy(col('timestamp').desc()) )

# COMMAND ----------


