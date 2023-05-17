# Databricks notebook source
from steps.config import *
from steps.evaluation import *



# COMMAND ----------

model_name1 = my_model_name 
model_version1 = model_version_to_submit
model_name2 = "sp_" + my_model_name
model_version2 = func_grab_newest_version_number_of_a_model(model_name2, stage = "None") #newest version of this model under this name

# COMMAND ----------

model1 = func_grab_model_based_on_name_and_version (model_name1, model_version1)
model2 = func_grab_model_based_on_name_and_version (model_name2, model_version2)

np.random.seed(0)
df_train = spark.table(train_data_set_name)
X_train = df_train.drop(*['y']).toPandas()
y_train = df_train.select(*['y']).toPandas()

accuracy_model1 = evaluate_accuracy(model1, X_train, y_train)
accuracy_model2 = evaluate_accuracy(model2, X_train, y_train)

# COMMAND ----------

val_diff_accuracy_betwee_2_models = abs((accuracy_model1 - accuracy_model2)/accuracy_model1)
if val_diff_accuracy_betwee_2_models<0.01:
  print("The difference is", val_diff_accuracy_betwee_2_models, ",test passed! ", "accuracy is about",accuracy_model1)
  print("User's model and version is:", model_name1, model_version1)
  print("SP's model and version is:", model_name2, model_version2)
  # update the description of the latest sp model version
  client.update_model_version(name=model_name2,version=model_version2, description="Tests passed, ready to transition to 'Production'")
  # transition the latest sp model version to 'Staging'
  client.transition_model_version_stage(name=model_name2, version=model_version2, stage="Staging")
  print("Just transitioned", model_name2, "last version", model_version2, "to 'Staging'")
else:
  print('accuracy_model1 user model', accuracy_model1)
  print('accuracy_model2 SP model', accuracy_model2)
  raise Exception ("error!!!!! DS trying to push version of production model that is not matching with Service Principal's latest model version")

# COMMAND ----------


