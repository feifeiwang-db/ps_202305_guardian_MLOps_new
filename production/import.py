# Databricks notebook source
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
original_workspace_name = func_grab_workspace_name()   

print("original_workspace_name:", original_workspace_name)
