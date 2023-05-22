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

def func_todaysdate():
  '''
  return todays date in string format, such as "20230329" in PST time zone
  '''
  from datetime import datetime, timedelta
  now_EST = datetime.today() - timedelta(hours=8)
  todaysdate = now_EST.strftime('%Y_%m_%d')
  todaysdate = todaysdate.replace("_", "")
  return todaysdate

def func_list_models_in_production(bool_print_info=False):
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

def func_should_export_this_production_version_or_not (model_name: str):
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
      print("Though this model:", model_name,"has latest Production stage in version",latest_production_version, ", it has previously already been exported. Thus, no actions here. If you still want to export this version again, please delete or modify the description of this version, and rerun code/pipeline")  
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
  dbutils.fs.mv(old_name, new_name, True) #rename that file path, so can copy later fine
  dbutils.fs.cp(output_dir, external_location, True)   
  print("Exported to external_location: ", external_with_run_id)    


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
#   list_info_workspace = eval(spark.conf.get("spark.databricks.clusterUsageTags.clusterAllTags"))
#   for i in range(0, len(list_info_workspace)):
#     dict_i = list_info_workspace[i]
#     if dict_i['key'] == 'dd_workspace':
#       return dict_i['value']  #in Azure
#   return "unknown"
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
todaysdate = func_todaysdate()
original_workspace_name = func_grab_workspace_name()   

# summary table related vars
dbutils.widgets.text("model_name_prefix", "") # by default, it should be blank
model_name = dbutils.widgets.get("model_name_prefix") + my_model_name

# model_name = "sp_"+ my_model_name #sp model, only admin and SP are having access
# model_name =  my_model_name #user controled, if user wants to run this export code

external_export_summary_location = "%s/export_summary_delta"% (external_location_folder)
external_export_summary_table_name = "export_summary.delta"
temp_internal_table_name = "table_full_summary"
bool_summary_table_exists_already, summary_table_location = func_summary_table_if_exists_already (external_export_summary_location, external_export_summary_table_name)

print("export summary_table_location: ", summary_table_location)
print("bool_summary_table_exists_already: ", bool_summary_table_exists_already)

# COMMAND ----------

# DBTITLE 1,Export
# list_models_having_production = func_list_models_in_production (bool_print_info=False) # find a list of models that have production stages
# print("list_models_having_production: ", list_models_having_production)
# assert model_name in list_models_having_production, "your model: "+model_name+" does not have a 'Production' stage, please manually transition the model stage from 'Staging' to 'Production' first" #optional additional testing, ensure engineers do not forget to transition the model to production
# ### note: for some reason, the SP does not see the "sp_feifei_model", but seeing all other models that are having production version

# make sure this model has at least one production version
assert len(func_check_model_having_production_versions(model_name, "Production"))>0, f"There is no Production stage in your model: {model_name}. Please manually transition the latest tests passed Staging model into 'Production' stage, and rerun the code/pipeline"
# store the export information into summary table
# 1. write latest summmary info into this internal table
if bool_summary_table_exists_already == True:
  df_summary = spark.read.load(path = summary_table_location)
  df_summary.write.mode("overwrite").saveAsTable(temp_internal_table_name)


output_dir = "%s/%s/%s/%s" % (output_dir_base, "workspace_"+original_workspace_name, todaysdate, model_name)
external_location = "%s/mlflowexportmodels/%s/%s/%s" % (external_location_folder, "workspace_"+original_workspace_name, todaysdate, model_name) # modify it to workspace/date/model_name level
print("external_location for model", external_location)

bool_should_export_this_production_version, latest_production_version, latest_description_production_version = func_should_export_this_production_version_or_not (model_name) 

# export the model
if bool_should_export_this_production_version == True:
  print("Working on exporting model_name:", model_name)

  print("Output temporary dbfs dir:", output_dir)
  run_id = func_grab_run_id_by_model_name_version(model_name, latest_production_version)
  external_with_run_id = "%s/%s" % (external_location, run_id)

  func_export(model_name, latest_production_version, notebook_formats, output_dir, external_location, run_id)
  # 1. after the export, make the description of the original model's latest production version as "Exported Already On......"
  client.update_model_version( 
  name=model_name,
  version=latest_production_version,
  description="Exported Already On " + todaysdate + ", old description: " + latest_description_production_version
  )
  # 2. construct the summary information from today about newly exported models
  df_summary_today = spark.createDataFrame(
  [
      (todaysdate,"workspace_"+original_workspace_name, model_name, latest_production_version, external_with_run_id),  # create your data here, be consistent in the types.
  ],
  ['todaysdate','from_workspace','exported_model_name','exported_model_version', 'external_location'] 
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
display(df_summary.orderBy(col('timestamp').desc() , col('todaysdate').desc(), col('from_workspace'), col('exported_model_name')) )

# COMMAND ----------


